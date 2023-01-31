import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hooks import newedge_hook
from models import OrthogMLP
from train import Trainer



device = torch.device("cpu")
DATA_SIZE = 60000


def main(train, test, N, loss_fc, lr, opt, regularization, epochs):
    path = r"C:\Users\Avery\Projects\ModularitySERI\saved_models\sai\mnist_784x256x64x32x10_SGD_LR1_reg0_trial000\mnist_784x256x64x32x10_SGD_LR1_reg0_trial000_epoch099.pt"

    # network = build_model()
    # network = train_model(train, test, N, loss_fc, lr, opt, regularization, epochs)
    network = load_model(path)

    gram = compute_gram(network, train)

    lam, S, U = eigensolve(gram)

    M = compute_M(network, gram, train)

    D, Dinvrs, V = eigensolve(M)

    new_edges = transform_network(network, train, U, S, V)

    redefined_model = change_edges(network, new_edges)

    # analyze(redefined_model)


def build_model():
    """ Build a model """
    model = OrthogMLP(4, 3, 2, 1)
    with torch.no_grad():
        model.fc0.weight = nn.Parameter(torch.ones_like(model.fc0.weight))
        model.fc1.weight = nn.Parameter(torch.tensor([[1., -2.], [3., -4.], [5., -6.]], requires_grad=True).T)
        model.fc2.weight = nn.Parameter(torch.tensor([[1., 1.]], requires_grad=True))
    return model


def train_model(train_loader, test_loader, N, loss_fc, lr, opt, regularization, epochs, save_path="", model_name=""):
    network = OrthogMLP(*N).to(device)
    trainer = Trainer(network, N, loss_fc, lr, opt, regularization, epochs, train_loader, test_loader, device, save_path, model_name)
    trainer.train()
    return network


def load_model(path):
    model = OrthogMLP(*N).to(device)
    model.load_state_dict(torch.load(path))
    return model


def compute_gram(model, dataloader):
    """ Orthogonalize the input model using the specified method """
    add_hooks(model, [model.grab_activations_hook])
    for b, (x, label) in enumerate(dataloader):
        model(x.reshape(len(x), -1).to(device))
        activations = model.activations
        if b == 0:
            grams = {i: torch.zeros((act.shape[1], act.shape[1])).to(device) for i, act in enumerate(activations)}
        for i, act in enumerate(activations):
            grams[i] += torch.matmul(act.transpose(0, 1), act)

    for k,v in grams.items():
        grams[k] = grams[k] / DATA_SIZE

    remove_hooks(model)
    return grams


def eigensolve(matrices: dict):
    """ Calculate the change of basis matrices for the orthogonalized network """
    eigenvalues, norm_eigenvalues, eigenvectors = {}, {}, {}
    for name, gram in matrices.items():
        lam, U = torch.linalg.eig(gram)
        eigenvalues[name] = torch.abs(lam)
        norm_eigenvalues[name] = torch.where(lam != 0, torch.pow(torch.abs(lam), -0.5), torch.tensor([0.0], device=lam.device))
        eigenvectors[name] = torch.abs(U)
    return eigenvalues, norm_eigenvalues, eigenvectors


def compute_M(model, grams, dataloader):
    M = {}

    add_hooks(model, hookfuncs=[model.compute_derivatives_hook])

    for b, (x, label) in enumerate(dataloader):
        model.derivatives = []
        model(x.reshape(len(x), -1).to(device))

        # Remove the connection to the phantom bias in the last layer todo this won't work if there is 1 output
        # model.derivatives[-1] = model.derivatives[-1][:,1].reshape(model.derivatives[-1].shape[0], 1)
        model.derivatives[-1] = model.derivatives[-1][:,:,1:]
        # dfs should be: (n+1)x(m+1) each layer but (n+1)x(m) last layer; n := L, m:= L+1;

        for n, df in enumerate(model.derivatives):
            gram = grams[n]
            M.setdefault(n, 0)
            # M[n] += df @ df.T @ gram @ gram.T + gram @ gram.T @ df @ df.T
            #       4x3   3x4   4x4     4x4     4x4     4x4     4x3   3x4
            # if batches:
            M[n] += torch.sum((df @ (df.permute(0,2,1) @ (gram @ gram.T))) + (gram @ (gram.T @ (df @ df.permute(0,2,1)))), dim=0)

    for k,v in M.items():
        M[k] = M[k] / DATA_SIZE  #todo change this to be # samples instead of batches

    remove_hooks(model)
    return M


def transform_network(model, dataloader, u, s, v):
    """ Calculate the new_edges of the model using the specified method """
    edges = {}
    add_hooks(model, [model.compute_derivatives_hook, model.grab_activations_hook])

    for b, (x, label) in enumerate(dataloader):
        model.derivatives = []
        model(x.reshape(len(x), -1).to(device))
        # model.derivatives[-1] = model.derivatives[-1][:,1].reshape(model.derivatives[-1].shape[0], 1)
        model.derivatives[-1] = model.derivatives[-1][:,:,1:]

        for l in range(len(model.layers)-1):
            # a = torch.diag(s[l + 1])
            # b = u[l + 1]
            # c = model.derivatives[l].T
            # d = u[l].T
            # e = torch.pow(torch.diag(s[l]), -1)

            s_d = torch.diag(s[l])
            s_invrse = torch.where(s_d != 0, torch.pow(s_d, -1), torch.tensor([1.0], device=s[l].device))

            # a = torch.diag(s[l+1]).shape
            # b_ = u[l+1].shape
            # c = model.derivatives[l].T.shape
            # d = u[l].T.shape
            # e = s_invrse.shape
            # ls = v[l+1] @ (torch.diag(s[l+1]) @ (u[l+1] @ model.activations[l+1].T))
            #    4x4             4x4                4x4        4x1
            # m = v[l+1] @ (torch.diag(s[l+1]) @ (u[l+1] @ (model.derivatives[l].T @ (u[l].T @ (s_invrse @ v[l].T)))))
            #    4x4           4x4               4x4             4x5                5x5          5x5     5x5
            # rs = v[l] @ (torch.diag(s[l]) @ (u[l] @ model.activations[l].T))
            #    5x5        5x5               5x5       5x1

            ls = v[l+1] @ (torch.diag(s[l+1]) @ (u[l+1] @ model.activations[l+1].T))
            m = v[l+1] @ (torch.diag(s[l+1]) @ (u[l+1] @ (model.derivatives[l].permute(0,2,1) @ (u[l].T @ (s_invrse @ v[l].T)))))
            rs = v[l] @ (torch.diag(s[l]) @ (u[l] @ model.activations[l].T))

            edges.setdefault(l, 0)
            value = torch.zeros_like(m[0])
            for i in range(len(m)):
                a = ls.T[i].reshape(1, ls.shape[0]).T
                b = m[i]
                c = rs.T[i].reshape(1, rs.shape[0])
                value += a * b * c
            # value = ls * m * rs.T if not batched
            #      4x1 * 4x5  * 1x5
            #     4x16 * 16x4x5 * 16x5
            edges[l] += value

        l += 1
        s_d = torch.diag(s[l])
        s_invrse = torch.where(s_d != 0, torch.pow(s_d, -1), torch.tensor([1.0], device=s[l].device))
        ls = (torch.diag(s[l + 1]) @ (u[l + 1] @ model.activations[l + 1].T))
        m = (torch.diag(s[l + 1]) @ (u[l + 1] @ (model.derivatives[l].permute(0,2,1) @ (u[l].T @ s_invrse))))
        rs = (torch.diag(s[l]) @ (u[l] @ model.activations[l].T))
        edges.setdefault(l, 0)
        value = torch.zeros_like(m[0])
        for i in range(len(ls)):
            a = ls.T[i].reshape(1, ls.shape[0]).T
            b = m[i]
            c = rs.T[i].reshape(1, rs.shape[0])
            value += a * b * c
        # value = ls * m * rs.T  if not batched
        edges[l] += value

    for k,v in edges.items():
        edges[k] = edges[k] / DATA_SIZE   # TODO change b

    remove_hooks(model)
    return edges


def change_edges(model, new_edges):
    """ Change the edges of the transformed model using the calculated edges """
    return model


def analyze(model):
    """ Analyze the redefined model and perform any necessary operations """
    pass


def intermediate_inspectoin(model, change_of_basis):
    """ Change the basis of the trained model using the change of basis matrix """
    # rotated_model = copy.deepcopy(model)
    for l in range(len(model.layers)):
        w = change_of_basis[l].T @ model.layers[l].weight @ change_of_basis[l]
        # model.layers[l].weight = nn.Parameter(torch.matmul(change_of_basis[l],
        #                                                    torch.matmul(model.layers[l].weight, change_of_basis[l].T)),
        #                                       requires_grad=True)
    return model


def add_hooks(model, hookfuncs: list):
    model.handles = []
    for module in model.layers:
        for hook in hookfuncs:
            handle = module.register_forward_hook(hook)
            model.handles.append(handle)


def remove_hooks(model):
    for h in model.handles:
        h.remove()


if __name__ == '__main__':
    x = torch.ones((1, 4), requires_grad=True)
    y = torch.tensor([42])
    data = [(x,y)]
    batch_size = 16
    loss_fc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD
    lr = .1
    N = [784, 256, 64, 32, 10]
    epochs = 100
    regularization = 0

    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=True,
                                                              transform=transforms.Compose([transforms.ToTensor(),
                                                                                            transforms.Normalize(
                                                                                                (0.1307,),
                                                                                                (0.3081,))])),
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=False,
                                                             transform=transforms.Compose([transforms.ToTensor(),
                                                                                           transforms.Normalize(
                                                                                               (0.1307,), (0.3081,))])),
                                              batch_size=1,
                                              shuffle=True)

    main(train_loader, test_loader, N, loss_fc, lr, optimizer, regularization, epochs)






