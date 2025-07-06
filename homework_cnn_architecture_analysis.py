import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from utils.training_utils import train_model, evaluate_model
from utils.visualization_utils import plot_learning_curves, plot_feature_maps
from utils.comparison_utils import count_parameters
from models.cnn_models import ResNetCNN


def load_cifar10(batch_size=128):
    """
    Тут я загружаю CIFAR-10 и подготавливаю загрузчики для тренировки и теста.
    Применяю преобразования: перевод в тензор и нормализацию по средним и отклонениям.

    :param batch_size: число примеров в одном батче
    :return: train_loader, test_loader
    """
    # Составляем цепочку преобразований для изображений
    transform = transforms.Compose([
        transforms.ToTensor(),  # переводим в тензор
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # нормализация
    ])
    # Скачиваем обучающий и тестовый датасеты
    train_ds = torchvision.datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.CIFAR10('data/cifar10', train=False, download=True, transform=transform)
    # Оборачиваем в DataLoader для батчей и перемешивания
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def compute_receptive_field(kernel_sizes):
    """
    Вычисляем рецептивное поле для последовательных conv-слоев со stride=1.
    Формула простая: rf = 1 + sum(kernel_size - 1).

    :param kernel_sizes: список размеров ядер свертки
    :return: размер рецептивного поля
    """
    # складываем (k-1) для каждого ядра и добавляем 1
    return 1 + sum(k - 1 for k in kernel_sizes)


class KernelCNN(nn.Module):
    """
    Простая CNN с двумя conv-слоями разных размеров ядер.
    Каждый conv -> BatchNorm -> ReLU, затем усредняем, "
    "выпрямляем" и финальный линейный слой для классификации.
    """
    def __init__(self, in_channels, num_classes, kernels):
        super().__init__()
        layers = []
        c = in_channels
        # создаем conv-блоки для каждого ядра
        for k in kernels:
            layers += [
                nn.Conv2d(c, 64, kernel_size=k, padding=k//2, bias=False),  # свертка
                nn.BatchNorm2d(64),  # нормируем
                nn.ReLU()  # нелинейность
            ]
            c = 64
        # после сверток усредняем по всем пространственным координатам
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # прямой проход через все слои
        return self.net(x)


def run_kernel_size_experiments(device):
    """
    Тут мы проверяем, как размер ядра влияет на обучение сети.
    Пробуем ядра [3,3], [5,5], [7,7] и комбинированный [1,3].
    Считаем параметры, рецептивное поле, время тренировки и точность.
    Строим графики и смотрим активации первого слоя.

    :param device: устройство (CPU или GPU)
    :return: словарь с результатами для каждого набора ядер
    """
    # загружаем данные
    train_loader, test_loader = load_cifar10()
    kernels_list = [[3, 3], [5, 5], [7, 7], [1, 3]]  # варианты ядер
    results = {}
    for ks in kernels_list:
        name = f"Kernels_{'x'.join(map(str, ks))}"
        print(f"\nЭксперимент: {name}")
        # создаем модель с текущими ядрами
        model = KernelCNN(3, 10, ks).to(device)
        # считаем сколько у нее параметров
        params = count_parameters(model)
        # вычисляем рецептивное поле
        rf = compute_receptive_field(ks)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)  # оптимизатор Adam
        criterion = nn.CrossEntropyLoss()  # функция потерь

        # тренируем и замеряем время
        start = time.time()
        history = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=10, device=device)
        train_time = time.time() - start

        # тестируем точность
        acc = evaluate_model(model, test_loader, device)
        # строим графики обучения
        plot_learning_curves(history, title=f"{name} Learning Curves")
        # визуализируем первые 8 карт признаков после первой свертки
        plot_feature_maps(
            model.net[0], next(iter(test_loader))[0].to(device), num_maps=8,
            title=f"{name} First-Layer Activations"
        )

        # сохраняем результаты
        results[name] = {'params': params, 'rf': rf, 'time': train_time, 'test_acc': acc}
    return results


class DepthCNN(nn.Module):
    """
    CNN, где мы сами настраиваем количество conv-слоев.
    Каждый слой: Conv3x3 -> ReLU, а в конце AvgPool + Flatten + Linear.
    """
    def __init__(self, in_channels, num_classes, num_layers):
        super().__init__()
        layers = []
        c = in_channels
        # добавляем нужное число слоев
        for _ in range(num_layers):
            layers += [nn.Conv2d(c, 64, 3, padding=1), nn.ReLU()]  # свертка + активация
            c = 64
        # усредняем, выпрямляем и классифицируем
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # прямой проход


def run_depth_experiments(device):
    """
    Тут мы исследуем влияние глубины сети: 2, 4 и 6 conv-слоев.
    Потом добавляем ResNetCNN для сравнения.
    Для каждой модели считаем точности на тренировке и тесте и строим графики.

    :param device: устройство (CPU или GPU)
    :return: словарь с точностями для каждого варианта
    """
    train_loader, test_loader = load_cifar10()
    depths = [2, 4, 6]  # варианты глубины
    results = {}
    for d in depths:
        name = f"Depth_{d}"
        print(f"\nЭксперимент: {name}")
        model = DepthCNN(3, 10, d).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # тренируем
        history = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=10, device=device)
        # считаем точности
        train_acc = evaluate_model(model, train_loader, device)
        test_acc = evaluate_model(model, test_loader, device)
        # показываем кривые обучения
        plot_learning_curves(history, title=f"{name} Learning Curves")
        results[name] = {'train_acc': train_acc, 'test_acc': test_acc}

    # сравнение с ResNetCNN
    print("\nЭксперимент: ResNetCNN")
    resnet = ResNetCNN().to(device)
    optimizer = optim.Adam(resnet.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    history = train_model(resnet, train_loader, test_loader, crit, optimizer, epochs=10, device=device)
    train_acc = evaluate_model(resnet, train_loader, device)
    test_acc = evaluate_model(resnet, test_loader, device)
    plot_learning_curves(history, title="ResNetCNN Learning Curves")
    results['ResNetCNN'] = {'train_acc': train_acc, 'test_acc': test_acc}

    return results


def main():
    """
    Главная точка входа: выбираем устройство и запускаем оба эксперимента.
    """
    # выбираем CPU или GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")

    print("=== Анализ размеров ядер ===")
    ks_results = run_kernel_size_experiments(device)

    print("\n=== Анализ глубины сети ===")
    depth_results = run_depth_experiments(device)

    print("Анализ завершен.")


if __name__ == '__main__':
    main()
