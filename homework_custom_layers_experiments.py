import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import time

class CustomConv2d(nn.Conv2d):
    """
    Кастомный сверточный слой:
    - Наследует nn.Conv2d
    - Добавляет learnable масштабирование и динамическую нормализацию по каналам
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))

    def forward(self, x):
        y = super().forward(x)
        mean = y.mean(dim=(0,2,3), keepdim=True)
        std = y.std(dim=(0,2,3), keepdim=True) + 1e-5
        y_norm = (y - mean) / std
        return self.scale * y_norm

class SpatialAttention(nn.Module):
    """
    Простой Spatial Attention:
    - Конкатенация channel-wise avg и max
    - 2D-свёртка + сигмоида
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3,7)
        padding = (kernel_size-1)//2
        self.conv = nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg = x.mean(dim=1,keepdim=True)
        maxv,_ = x.max(dim=1,keepdim=True)
        attn = torch.cat([avg,maxv],dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn

class CustomActivation(Function):
    """
    Кастомная активация: f(x)=x*sigmoid(x)
    Реализация forward и backward
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input * torch.sigmoid(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sig = torch.sigmoid(input)
        grad = grad_output * (sig * (1 + input*(1-sig)))
        return grad

class CustomActLayer(nn.Module):
    """Обёртка для CustomActivation"""
    def forward(self,x):
        return CustomActivation.apply(x)

class AdaptiveLpPool2d(nn.Module):
    """
    Глобальный Lp-пулинг:
    - Среднее значение abs(x)^p по spatial
    - Возвращает (mean)^(1/p)
    """
    def __init__(self,p=2):
        super().__init__()
        self.p = p

    def forward(self,x):
        return (x.abs().pow(self.p).mean(dim=[2,3])).pow(1.0/self.p)

# Тестирование кастомных слоев

def test_custom_layers():
    x = torch.randn(2,3,32,32)
    layers = [CustomConv2d(3,8,3,padding=1), SpatialAttention(), CustomActLayer(), AdaptiveLpPool2d(p=3)]
    out = x
    for layer in layers:
        out = layer(out)
        print(f"{layer.__class__.__name__} output shape: {out.shape}")

# === 3.2 Эксперименты с Residual-блоками ===

class BasicResidualBlock(nn.Module):
    """Basic Residual Block: conv-BN-ReLU -> conv-BN + skip + ReLU"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels,channels,3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels,channels,3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class BottleneckBlock(nn.Module):
    """Bottleneck Block: 1x1 -> 3x3 -> 1x1 conv"""
    def __init__(self,in_channels,bottleneck_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,bottleneck_channels,1,bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels,bottleneck_channels,3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels,in_channels,1,bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return self.relu(out)

class WideResidualBlock(nn.Module):
    """Wide Residual Block: ширина увеличена в середине"""
    def __init__(self,in_channels,widen_factor=2):
        super().__init__()
        mid = in_channels * widen_factor
        self.conv1 = nn.Conv2d(in_channels,mid,3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid,in_channels,3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

# Сравнение блоков: количество параметров и стабильность обучения
def compare_residual_blocks(device):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    ds = torchvision.datasets.CIFAR10('data',train=True,download=True,transform=transform)
    loader = DataLoader(ds,batch_size=64,shuffle=True)
    blocks = {
        'Basic': BasicResidualBlock(3),
        'Bottleneck': BottleneckBlock(3,16),
        'Wide': WideResidualBlock(3,4)
    }
    results = {}
    for name,blk in blocks.items():
        model = nn.Sequential(blk, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3 if name=='Basic' else (3 if name=='Wide' else 3),10)).to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.train()
        imgs,targets = next(iter(loader))
        imgs,targets = imgs.to(device),targets.to(device)
        start = time.time()
        out = model(imgs)
        loss = nn.CrossEntropyLoss()(out,targets)
        loss.backward()
        t = time.time()-start
        results[name] = {'params':params,'time_first_pass':t}
    return results

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=== Testing custom layers ===")
    test_custom_layers()
    print("=== Comparing residual blocks ===")
    print(compare_residual_blocks(device))
