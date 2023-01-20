import os
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from analysis import interactive_histogram
from eigen import compute_eigens
from gram import compute_grams, preprocess_lams
from models import OrthogMLP


def load_models(model_dir, device, N, which_models):
    model_paths = [os.path.join(model_dir, path) for path in os.listdir(model_dir) if path.endswith('.pt')]
    sorted_paths = sorted(model_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    models = []
    for path in sorted_paths:
        model = OrthogMLP(784, *N).to(device)
        model.load_state_dict(torch.load(path))
        models.append(model)
    return [models[i] for i in which_models]


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


def compute_gram_eigs(models: list, dataloader, N, per_layer, device):
    Gram_eigs = []

    print("Computing Grams")
    for network in models:
        if per_layer:
            grams = compute_grams(network, dataloader, True, device)
            U, lam = compute_eigens(grams)
            lam = preprocess_lams(lam, N)
        else:
            grams = compute_grams(network, dataloader, False, device)
            U, lam = compute_eigens(grams)

        Gram_eigs.append(lam)
    return Gram_eigs


if __name__ == "__main__":
    batch_size = 1024
    device = torch.device("cpu")
    loss_fc = torch.nn.CrossEntropyLoss()
    N = [64, 64, 10]

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=False,
                                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                       transforms.Normalize((0.1307,), (0.3081,))])),
                                                                batch_size=batch_size,
                                                                shuffle=True)

    models = load_models("saved_models/512_256_64_SGD/", device, N, these_models)
    scores = evaluate_models(models, loss_fc, test_loader, device)
    eigs = compute_gram_eigs(models, test_loader, N, True, device)
    interactive_histogram(list_of_eigs, list_of_scores, these_models)








