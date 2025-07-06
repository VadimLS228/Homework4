import torch
import torch.nn as nn

class FCNet(nn.Module):
    """
    Полносвязная сеть с возможностью "глубокой" версии через параметр deep.
    Использует LazyLinear, чтобы автоматически определять размер входа.
    """
    def __init__(self, deep: bool = False, num_classes: int = 10):
        super(FCNet, self).__init__()
        layers = [nn.Flatten()]
        if deep:
            # Глубокая конфигурация
            layers += [
                nn.LazyLinear(512), nn.ReLU(),
                nn.LazyLinear(256), nn.ReLU(),
                nn.LazyLinear(128), nn.ReLU()
            ]
        else:
            # Базовая конфигурация
            layers += [
                nn.LazyLinear(256), nn.ReLU(),
                nn.LazyLinear(128), nn.ReLU()
            ]
        layers.append(nn.LazyLinear(num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)