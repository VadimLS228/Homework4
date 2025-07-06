import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Простая CNN: 2 Conv слоя + полносвязный классификатор.
    Предназначена для одноканальных изображений (MNIST).
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Для MNIST: после двух пулов размер 7x7, 64 каналов
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """
    Обычный residual block с двумя conv-слоями и проекцией при необходимости.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNetCNN(nn.Module):
    """
    CNN с residual-блоками. Опционально можно включить дропаут для регуляризации.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 10, use_regularization: bool = False):
        super(ResNetCNN, self).__init__()
        self.use_reg = use_regularization
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if self.use_reg:
            self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        layers = [ResidualBlock(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        if self.use_reg:
            x = self.dropout(x)
        return self.fc(x)
