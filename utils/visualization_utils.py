import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import math

sns.set()

def plot_learning_curves(history, title=None):
    """
    Строит кривые потерь и точности для train и val из history.
    """
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.legend()
    if title:
        plt.title(title)
    plt.show()


def plot_confusion_matrix(model, loader, device, classes, title=None):
    """
    Строит матрицу ошибок для модели на loader.
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.numpy())
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    if title:
        plt.title(title)
    plt.show()


def plot_feature_maps(layer, input_tensor, num_maps=8, title=None):
    """
    Визуализирует feature maps, полученные после применения слоя к input_tensor.

    :param layer: nn.Module (например, первый Conv слой модели)
    :param input_tensor: torch.Tensor батч изображений [N, C, H, W]
    :param num_maps: число каналов, которые нужно отобразить
    :param title: заголовок графика
    """
    # Переводим слой в режим eval и отключаем градиенты
    layer.eval()
    with torch.no_grad():
        # Прогоняем первый пример батча через слой
        input_img = input_tensor[0].unsqueeze(0).to(next(layer.parameters()).device)
        activations = layer(input_img)
        # Переносим на CPU и извлекаем карту активаций [channels, H, W]
        maps = activations.cpu().squeeze(0)

    # Подготовка сетки для визуализации
    n_cols = min(4, num_maps)
    n_rows = math.ceil(num_maps / n_cols)
    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    for idx in range(num_maps):
        if idx >= maps.shape[0]:
            break
        plt.subplot(n_rows, n_cols, idx + 1)
        plt.imshow(maps[idx], cmap='viridis')
        plt.axis('off')
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()