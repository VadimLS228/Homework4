import time
import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    """
    Обучает модель и возвращает историю обучения.

    :param model: nn.Module
    :param train_loader: DataLoader для обучения
    :param val_loader: DataLoader для валидации/теста
    :param criterion: функция потерь
    :param optimizer: оптимизатор
    :param epochs: число эпох
    :param device: устройство ('cpu' или 'cuda')
    :return: dict с ключами 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            _, preds = torch.max(output, 1)
            correct += (preds == target).sum().item()
            total += target.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                _, preds = torch.max(output, 1)
                val_correct += (preds == target).sum().item()
                val_total += target.size(0)
        history['val_loss'].append(val_loss / val_total)
        history['val_acc'].append(val_correct / val_total)
        print(f"Epoch {epoch}/{epochs} - "
              f"Train loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f} - "
              f"Val loss: {val_loss/val_total:.4f}, acc: {val_correct/val_total:.4f}")
    return history


def evaluate_model(model, loader, device):
    """
    Вычисляет точность модели на переданном DataLoader.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / total


def measure_inference_time(model, loader, device, num_batches: int = 10):
    """
    Измеряет среднее время инференса по числу батчей.
    """
    model.eval()
    timings = []
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            if i >= num_batches:
                break
            data = data.to(device)
            start = time.time()
            _ = model(data)
            timings.append(time.time() - start)
    return sum(timings) / len(timings)
