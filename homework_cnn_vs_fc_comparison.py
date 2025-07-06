import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from models.fc_models import FCNet  # 3-4 layer fully-connected network
from models.cnn_models import SimpleCNN, ResNetCNN  # простая CNN и CNN с residual-блоками
from utils.training_utils import train_model, evaluate_model, measure_inference_time
from utils.visualization_utils import plot_learning_curves, plot_confusion_matrix
from utils.comparison_utils import count_parameters, analyze_gradient_flow


def load_mnist(batch_size: int = 64):
    """
    Загружает датасет MNIST и готовит DataLoader для обучения и тестирования.

    Здесь я использую torchvision.datasets.MNIST, применяю преобразования:
    - ToTensor для перевода в тензор
    - Normalize для стандартизации пикселей

    :param batch_size: размер батча для загрузчика
    :return: train_loader, test_loader
    """
    # Описываю последовательность преобразований для изображений
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Скачиваем и создаем датасет для обучения
    train_dataset = torchvision.datasets.MNIST(
        root='data/mnist', train=True, download=True, transform=transform
    )
    # Скачиваем и создаем датасет для тестирования
    test_dataset = torchvision.datasets.MNIST(
        root='data/mnist', train=False, download=True, transform=transform
    )
    # Оборачиваем датасеты в DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_cifar10(batch_size: int = 64):
    """
    Загружает датасет CIFAR-10 и готовит DataLoader.

    Я применяю нормализацию по средним и стандартным отклонениям каналов RGB,
    чтобы улучшить сходимость обучения.

    :param batch_size: размер батча
    :return: train_loader, test_loader
    """
    # Стандартные параметры нормализации для CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root='data/cifar10', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='data/cifar10', train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def run_mnist_comparison(device: torch.device):
    """
    Запускает эксперимент на MNIST: сравнение Fully-Connected, SimpleCNN и ResNetCNN.

    :param device: устройство для обучения (cpu или cuda)
    :return: словарь с результатами (время, точность, инференс и параметры)
    """
    # Задаю основные гиперпараметры
    epochs = 10
    lr = 1e-3
    batch_size = 128

    # Готовлю загрузчики данных
    train_loader, test_loader = load_mnist(batch_size)

    # Определяю модели для сравнения
    models = {
        'FCNet': FCNet(),
        'SimpleCNN': SimpleCNN(),
        'ResNetCNN': ResNetCNN()
    }

    results = {}
    # Прохожусь по всем моделям и обучаю каждую
    for name, model in models.items():
        print(f"Начинаю обучение {name} на MNIST...")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Измеряю время тренировки
        start_time = time.time()
        history = train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device)
        training_time = time.time() - start_time

        # Оцениваю точность на тестовой выборке
        test_acc = evaluate_model(model, test_loader, device)
        # Измеряю время инференса
        inference_time = measure_inference_time(model, test_loader, device)

        # Считаю количество параметров в модели
        params = count_parameters(model)

        # Строю графики кривых обучения
        plot_learning_curves(history, title=f"{name} Learning Curves (MNIST)")

        # Сохраняю результаты в словарь
        results[name] = {
            'train_time': training_time,
            'test_accuracy': test_acc,
            'inference_time': inference_time,
            'num_params': params,
            'history': history
        }
    return results


def run_cifar_comparison(device: torch.device):
    """
    Запускает эксперимент на CIFAR-10: сравнение DeepFC, ResNetCNN и ResNetCNN с регуляризацией.

    :param device: устройство для обучения
    :return: словарь с результатами (точность и история обучения)
    """
    # Гиперпараметры эксперимента
    epochs = 20
    lr = 1e-3
    batch_size = 128

    # Загружаю CIFAR-10
    train_loader, test_loader = load_cifar10(batch_size)

    # Готовлю модели: глубокая FC сеть, CNN с residual и с регуляризацией
    models = {
        'DeepFC': FCNet(deep=True),
        'ResNetCNN': ResNetCNN(),
        'ResNetCNN_Reg': ResNetCNN(use_regularization=True)
    }

    results = {}
    for name, model in models.items():
        print(f"Начинаю обучение {name} на CIFAR-10...")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Обучаю модель и получаю историю метрик
        history = train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device)

        # Оцениваю точности на train и test
        train_acc = evaluate_model(model, train_loader, device)
        test_acc = evaluate_model(model, test_loader, device)

        # Строю матрицу ошибок
        plot_confusion_matrix(
            model, test_loader, device,
            classes=test_loader.dataset.classes,
            title=f"{name} Confusion Matrix (CIFAR-10)"
        )

        # Анализ градиентного потока, чтобы понять, есть ли проблемы с затухающими/взрывающимися градиентами
        analyze_gradient_flow(model, title=f"{name} Gradient Flow (CIFAR-10)")

        # Сохраняю результаты
        results[name] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'history': history
        }
    return results


def main():
    """
    Основная точка входа для запуска всех сравнений.

    Определяю устройство, запускаю MNIST и CIFAR-10 эксперименты.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Использую устройство: {device}")

    print("\n--- Сравнение на MNIST ---")
    mnist_results = run_mnist_comparison(device)

    print("\n--- Сравнение на CIFAR-10 ---")
    cifar_results = run_cifar_comparison(device)

    # Здесь можно добавить дополнительные таблицы или анализ
    print("Все эксперименты завершены.")


if __name__ == '__main__':
    main()
