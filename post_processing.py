import os
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from analysis import plot_valid_losses
from eigen import compute_eigens
from gram import compute_grams, preprocess_lams
from models import OrthogMLP


def load_models(model_dir, device, N):
    model_paths = [os.path.join(model_dir, path) for path in os.listdir(model_dir) if path.endswith('.pt')]
    sorted_paths = sorted(model_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    models = []
    for path in sorted_paths:
        model = OrthogMLP(784, *N).to(device)
        model.load_state_dict(torch.load(path))
        models.append(model)
    return models


def evaluate_models(models, loss_func, test_loader, device):
    scores = []
    for m in range(len(models)):
        score = validation_fn(models[m], test_loader, loss_func, device)
        scores.append(score)
        print(f'Validation score for Model {m} is {score}')

    plt.plot(scores)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.show()


def validation_fn(model, test_loader, loss_func, device):
    # score = model.get_loss(test_loader, loss_func, device) / len(test_loader)
    score = model.get_accuracy(test_loader, device)
    return score


if __name__ == "__main__":
    batch_size = 1024
    device = torch.device("cpu")
    loss_fc = torch.nn.CrossEntropyLoss()
    N = [64, 64, 10]
    load_and_evaluate_models("saved_models/", loss_fc, device, N, batch_size)
