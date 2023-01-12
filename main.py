import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import _stateless

from analysis import plot_magnitude_frequency, preprocess_grams, plot_magnitude_frequency_by_layer,\
    preprocess_lams_full_network, repeat_and_concatenate, plot_hessians
from datasets import RetinaDataset
from models import OrthogMLP


conf_path = os.getcwd()
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main(batch_size):
    device = torch.device("cpu")

    # Train Model
    beltalowda = RetinaDataset(8)
    dataLoader = torch.utils.data.DataLoader(beltalowda, batch_size=batch_size)
    network_bad = OrthogMLP(8, 8, 4, 2, 1).to(device)
    network_good = OrthogMLP(8, 8, 4, 2, 1).to(device)
    loss_fc = torch.nn.MSELoss()
    opt_bad = torch.optim.Adam(lr=1e-3, params=network_bad.parameters())
    opt_good = torch.optim.Adam(lr=1e-3, params=network_good.parameters())
    goal_and = 1
    epochs1 = 500
    epochs2 = 2000
    print("Training Bad Model...")
    for i in range(epochs1):
        for b, (x, label) in enumerate(dataLoader):
            x = x.to(device)
            _and, _or = label
            result = _and if goal_and else _or
            prediction = network_bad(x)
            opt_bad.zero_grad()
            loss1 = loss_fc(prediction.view(-1), result.float().to(device))
            print("Loss: ", loss1)
            loss1.backward()
            opt_bad.step()
    print("Training Good Model...")
    for i in range(epochs2):
        for b, (x, label) in enumerate(dataLoader):
            x = x.to(device)
            _and, _or = label
            result = _and if goal_and else _or
            prediction = network_good(x)
            opt_good.zero_grad()
            loss2 = loss_fc(prediction.view(-1), result.float().to(device))
            print("Loss: ", loss2)
            loss2.backward()
            opt_good.step()

    print("Final Loss of less optimal model: ", loss1)
    print("Final Loss of more optimal model: ", loss2)

    # Gram Matrix
    # print("Finished Training. \nComputing Gram Matrix")
    # grams = compute_grams(network, dataLoader, per_layer=False)
    # U, lam = compute_eigens(grams)
    # check_eigens(grams, U)
    # ortho_layers = orthogonalize(network, U)

    # Hessian
    print("Computing Hessian")
    h, h_u, h_lam = compute_hessian(network_bad, dataLoader, loss_fc, goal_and)
    h2, h_u2, h_lam2 = compute_hessian(network_good, dataLoader, loss_fc, goal_and)

    # Analysis
    print("Plotting Frequencies")
    N = [8, 4, 2, 1]  # TODO hardcoded
    # lam = preprocess_grams(lam, N)
    # lam = preprocess_lams_full_network(lam, N) # Use this for norming across entire network
    # lam = repeat_and_concatenate(lam[0], N)
    # plot_magnitude_frequency(lam[0].detach(), h_lam)
    # plot_magnitude_frequency(lam, h_lam)
    plot_hessians(h_lam, h_lam2)
    # plot_magnitude_frequency_by_layer(lam, h_lam)


def compute_grams(model, dataloader, per_layer=False):
    Grams = []
    for b, (x, label) in enumerate(dataloader):  # TODO we probably want a different dataLoader for this step
        prediction = model(x)
        activations = model.activations
        if per_layer:
            for layer in activations:
                gram = torch.matmul(layer.transpose(0, 1), layer)
                gram = gram / 256
                Grams.append(gram)
        else:
            f = torch.cat(activations, dim=1)
            gram = torch.matmul(f.transpose(0, 1), f) / 256
            Grams.append(gram)
    return Grams


def compute_eigens(matrices: list):
    eigenvalues, eigenvectors = [],[]
    for N in matrices:
        evs, eigs = torch.linalg.eigh(N)
        eigenvalues.append(evs)
        eigenvectors.append(eigs)
    return eigenvectors, eigenvalues


def orthogonalize(model, U):
    ortho_layers = []
    for i in range(len(model.layers)):
        ortho_layers.append(torch.matmul(model.layers[i].weight, U[i].transpose(0,1)))
    return ortho_layers


def check_eigens(grams, U):
    for l in range(len(U)):
        check = torch.matmul(torch.matmul(U[l].transpose(0,1), grams[l]), U[l])
        print(f"Layer: {l} Is Diagonal: {is_diagonal(check)}")


def is_diagonal(A):
    # Create a matrix of the same shape as A with diagonal elements equal to the elements of A
    D = torch.diagflat(A.diag())
    # Check if the off-diagonal elements are zero
    off_diag = (D - A).abs().sum()
    # return off_diag == 0
    return off_diag


def compute_hessian(model, dataLoader, loss_func, goal_and):
    device = torch.device("cpu")

    all_batches_x = []
    all_batches_labels = []

    for b, (x, label) in enumerate(dataLoader):
        if b > 1:  # TODO I think I need to remove this
            break
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


if __name__ == "__main__":
    batch_size = 256
    main(batch_size)



