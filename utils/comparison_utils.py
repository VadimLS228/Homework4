import torch

def count_parameters(model):
    """
    Подсчитывает количество обучаемых параметров в модели.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_gradient_flow(model, title=None):
    """
    Извлекает и строит график градиентного потока для каждого слоя.
    """
    ave_grads = []
    max_grads = []
    layers = []
    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item() if param.grad is not None else 0)
            max_grads.append(param.grad.abs().max().item() if param.grad is not None else 0)
    # Построение графика
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(ave_grads, label='Avg Gradient')
    plt.plot(max_grads, label='Max Gradient')
    plt.xticks(range(len(ave_grads)), layers, rotation='vertical')
    plt.xlabel('Layers')
    plt.ylabel('Gradient value')
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
