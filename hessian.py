import torch
from torch.nn.utils import _stateless
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize

from manual_broadness import BroadnessMeasurer


def show_hessian(model, dataLoader, loss_func, goal_and):
    device = torch.device("cpu")

    all_batches_images = []
    all_batches_labels = []

    # for idx, (images, labels) in enumerate(train_loader):
    #     if idx > 1:
    #         break
    #     all_batches_images.append(images)
    #     all_batches_labels.append(labels)

    for b, (x, label) in enumerate(dataLoader):
        if b > 1:
            break
        x = x.to(device)
        _and, _or = label
        result = _and if goal_and else _or
        result = result.type(torch.float).to(device)

        all_batches_images.append(x)
        all_batches_labels.append(result)

    all_batches_images = torch.cat(all_batches_images, dim=0)#.cuda()
    all_batches_labels = torch.cat(all_batches_labels, dim=0)#.cuda()

    # Accumulate batches
    # def calculate_loss_function(*params):
    #     names = list(n for n, _ in model.named_parameters())
    #     preds = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, all_batches_images)
    #     return loss_fn(preds, all_batches_labels)

    # Accumulate batches
    def calculate_loss_function(*params):
        names = list(n for n, _ in model.named_parameters())
        preds = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, all_batches_images)
        return loss_func(preds, all_batches_labels)

    H = torch.autograd.functional.hessian(calculate_loss_function, tuple(model.parameters()))

    rows = []
    shapes = [p.shape for p in model.parameters()]
    for i in range(len(H)):
        rows.append(torch.cat([H[j][i].view(shapes[j].numel(), shapes[i].numel()) for j in range(len(H))], dim=0))

    full_hessian = torch.cat(rows, dim=1)

    # # Merge the hessians together
    # row_1_1 = H[0][0].flatten(0, 3).flatten(1,4)
    # row_1_2 = H[0][1].flatten(0, 3).flatten(1,4)
    # row_1_3 = H[0][2].flatten(0, 3).flatten(1,2)
    # row_1 = torch.cat([row_1_1, row_1_2, row_1_3], dim=1)
    #
    # row_2_1 = H[1][0].flatten(0, 3).flatten(1,4)
    # row_2_2 = H[1][1].flatten(0, 3).flatten(1,4)
    # row_2_3 = H[1][2].flatten(0, 3).flatten(1,2)
    # row_2 = torch.cat([row_2_1, row_2_2, row_2_3], dim=1)
    #
    # row_3_1 = H[2][0].flatten(0, 1).flatten(1,4)
    # row_3_2 = H[2][1].flatten(0, 1).flatten(1,4)
    # row_3_3 = H[2][2].flatten(0, 1).flatten(1,2)
    # row_3 = torch.cat([row_3_1, row_3_2, row_3_3], dim=1)
    #
    # full_hessian = torch.cat([row_1, row_2, row_3], dim=0)

    eigenvalues, eigenvectors = torch.linalg.eigh(full_hessian)
    eigenvalues = eigenvalues.float()

    eigenlist = eigenvalues.detach().cpu().numpy()

    plt.hist(eigenlist, bins=100)
    # plt.show()
    return eigenvectors, eigenvalues, eigenvectors.transpose(0,1)


def compute_hessian(model, dataLoader, loss_func, goal_and):
    device = torch.device("cpu")

    all_batches_x = []
    all_batches_labels = []

    for b, (x, label) in enumerate(dataLoader):
        # if b > 1:  # TODO I removed this
        #     break
        x = x.to(device)
        _and, _or = label
        result = _and if goal_and else _or
        result = result.type(torch.float).to(device)

        all_batches_x.append(x)
        all_batches_labels.append(result)

    all_batches_x = torch.cat(all_batches_x, dim=0)#.cuda()
    all_batches_labels = torch.cat(all_batches_labels, dim=0)#.cuda()

    # Accumulate batches
    def calculate_loss_function(*params):
        names = list(n for n, _ in model.named_parameters())
        preds = _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, all_batches_x)
        return loss_func(preds, all_batches_labels)

    H = torch.autograd.functional.hessian(calculate_loss_function, tuple(model.parameters()))

    rows = []
    shapes = [p.shape for p in model.parameters()]
    for i in range(len(H)):
        rows.append(torch.cat([H[j][i].view(shapes[j].numel(), shapes[i].numel()) for j in range(len(H))], dim=0))

    full_hessian = torch.cat(rows, dim=1)

    eigenvalues, eigenvectors = torch.linalg.eigh(full_hessian)
    eigenvalues = eigenvalues.float()

    return full_hessian, eigenvectors, eigenvalues


def newton_approximate(model, loss_fn, dataloader, device):
    all_batches_x, all_batches_labels = model.get_x_y_batches(dataloader, device)

    def closure(x, y):
        model.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        return loss

    # Optimize using BFGS
    # params = list(model.parameters())
    # res = minimize(closure, params, method='BFGS')

    # Optimize using L-BFGS-B
    params = list(model.parameters())
    res = minimize(closure, params, method='L-BFGS-B')
    hessian = res.x


def manual_approximation(model, loss_fc, dataloader, device):
    # Construct broadness measurer
    broadness = BroadnessMeasurer(model, dataloader, loss_fc)

    # Get starting loss
    starting_loss = model.get_loss(dataloader, loss_fc, device)

    # RUN
    print("Running Broadness Measurer")
    losses, deltas = broadness.run(std_list=[.001], num_itrs=2, normalize=False)

    # Average
    mean_losses = [np.mean(l) for l in losses]
    diffs = [ml - starting_loss for ml in mean_losses]
    mean_loss_across_stds = np.mean(mean_losses)

    return mean_losses

