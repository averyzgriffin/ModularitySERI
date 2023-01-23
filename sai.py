import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import OrthogMLP
from train import Trainer


def new_models():
    dimensions = [3, hidden_layers, num_labels]
    model = OrthogMLP(dimensions)
    trainer = Trainer(model, N, loss_fc, epochs, dataloader, test_loader, device)
    trainer.train()


def grab_activations(model, dataloader, device):
    for b, (x, label) in enumerate(dataloader):
        prediction = model(x.reshape(len(x), -1).to(device))
        activations = model.activations
        # TODO do your thing here


def load_models(model_dir, device, N):
    model_paths = [os.path.join(model_dir, path) for path in os.listdir(model_dir) if path.endswith('.pt')]
    # sorted_paths = sorted(model_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))  #  Ignore this
    models = []
    for path in model_paths:
        model = OrthogMLP(784, *N).to(device)
        model.load_state_dict(torch.load(path))
        models.append(model)
    return models


if __name__ == "__main__":
    path = ""
    batch_size = 100
    device = torch.device("cpu")
    loss_fc = torch.nn.CrossEntropyLoss()
    epochs = 100
    N = [64, 64, 10]  # Don't include 784 if using my models

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=False,
                                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                                           transforms.Normalize(
                                                                                               (0.1307,), (0.3081,))])),
                                              batch_size=batch_size,
                                              shuffle=True)

    models = load_models(path, device, N)
    model = models[0]  # TODO Use this if you don't want to look at all models at once
    grab_activations(model, test_loader, device)



