import os
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import OrthogMLP


def load_and_evaluate_models(model_dir, loss_func, device, N, batch_size):
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=False,
                                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                       transforms.Normalize((0.1307,), (0.3081,))])),
                                                            batch_size=batch_size,
                                                            shuffle=True)

    model_paths = [os.path.join(model_dir, path) for path in os.listdir(model_dir) if path.endswith('.pt')]
    sorted_paths = sorted(model_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    losses = []
    for path in sorted_paths:
        model = OrthogMLP(784, *N).to(device)
        model.load_state_dict(torch.load(path))
        loss = validation_fn(model, test_loader, loss_func, device)
        losses.append(loss)
        print(f'Validation loss for {path} is {loss}')
    # Plot the losses
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.show()


def validation_fn(model, test_loader, loss_func, device):
    loss = model.get_loss(test_loader, loss_func, device)
    normalized_loss = loss / len(test_loader)
    return normalized_loss


if __name__ == "__main__":
    batch_size = 1024
    device = torch.device("cpu")
    loss_fc = torch.nn.CrossEntropyLoss()
    N = [64, 64, 10]
    load_and_evaluate_models("saved_models/", loss_fc, device, N, batch_size)
