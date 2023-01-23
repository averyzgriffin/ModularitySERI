import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from analysis import interactive_histogram
from eigen import compute_eigens
from gram import compute_grams, preprocess_lams
from models import OrthogMLP


def load_models(model_dir, device, N, which_models):
    model_paths = [os.path.join(model_dir, path) for path in os.listdir(model_dir) if path.endswith('.pt')]
    sorted_paths = sorted(model_paths, key=lambda x: int(x.split("_")[-1].split(".")[0][-3:]))
    models = []
    for path in sorted_paths:
        model = OrthogMLP(*N).to(device)
        model.load_state_dict(torch.load(path))
        models.append(model)
    return [models[i] for i in which_models] if which_models else models


def evaluate_models(models, loss_func, test_loader, device):
    scores = []
    for m in range(len(models)):
        score = validation_fn(models[m], test_loader, loss_func, device)
        scores.append(score)
        print(f'Validation score for Model {m} is {score}')
    return scores


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
            lam = preprocess_lams(lam, N[2:])
        else:
            grams = compute_grams(network, dataloader, False, device)
            U, lam = compute_eigens(grams)

        Gram_eigs.append(lam)
    return Gram_eigs


def create_model_name(task, optimizer, lr, N, regularization):
    optimizer_name = optimizer.__name__
    lr_str = str(lr).replace("0.", "")
    N_str = "x".join(str(n) for n in N)
    name = f"{task}_{N_str}_{optimizer_name}_LR{lr_str}_reg{regularization}"
    return name


if __name__ == "__main__":
    batch_size = 1024
    device = torch.device("cpu")
    loss_fc = torch.nn.CrossEntropyLoss()
    task = "mnist"
    optimizer = torch.optim.SGD
    lr = .1
    Ns = [[784, 32, 32, 32, 10],[784, 32, 32, 32, 32, 32, 32, 10], [784, 512, 10], [784, 64, 10]]
    epochs = 100
    num_trials = 10
    regularization = 0
    n_bins = [100, 1000, 10000]
    these_models = list(range(0,10,1)) + list(range(10,100,5))
    if these_models == "all":
        these_models = list(range(0, epochs, 1))
    github_path = r"C:\Users\avery\Projects\github.io\averyzgriffin.github.io\gram_eigens".replace("\\", "/")

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=False,
                                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                                           transforms.Normalize((0.1307,), (0.3081,))])),
                                              batch_size=batch_size,
                                              shuffle=True)
    for N in Ns:
        for i in range(num_trials):
            model_name = create_model_name(task, optimizer, lr, N, regularization)
            load_path = f"saved_models/{task}/{model_name}/{model_name}_trial{str(i).zfill(3)}"
            models = load_models(load_path, device, N, these_models)

            scores = evaluate_models(models, loss_fc, test_loader, device)
            eigs = compute_gram_eigs(models, test_loader, N, True, device)

            save_path = f"{github_path}/{task}/{model_name}/{str(i).zfill(3)}"
            os.makedirs(save_path, exist_ok=True)
            interactive_histogram(eigs, scores, these_models, n_bins, save_path)

