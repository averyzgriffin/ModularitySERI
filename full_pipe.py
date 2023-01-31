import copy
import torch
from torch import nn

from hooks import newedge_hook
from models import OrthogMLP


device = torch.device("cpu")


def main(data):

    network = build_model()

    gram = compute_gram(network, data)

    lam, S, U = eigensolve(gram)

    # derivatives = get_derivatives(network, data)  # TODO I think it makes more sense to just compute this on the fly

    M = compute_M(network, gram, data)

    D, Dinvrs, V = eigensolve(M)

    new_edges = transform_network(network, data, U, S, V)

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


def compute_gram(model, dataloader):
    """ Orthogonalize the input model using the specified method """
    add_hooks(model, [model.grab_activations_hook])
    for b, (x, label) in enumerate(dataloader):
        model(x.reshape(len(x), -1).to(device))
        activations = model.activations
        if b == 0:
            grams = {i: torch.zeros((act.shape[1], act.shape[1])) for i, act in enumerate(activations)}
        for i, act in enumerate(activations):
            grams[i] += torch.matmul(act.transpose(0, 1), act)

    for k,v in grams.items():
        grams[k] = grams[k] / (b + 1)

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
        if b == 10:
            break
        model.derivatives = []
        model(x.reshape(len(x), -1).to(device))

        for n, df in enumerate(model.derivatives):
            gram = grams[n]
            M.setdefault(n, 0)
            M[n] += df @ df.T @ gram @ gram.T + gram @ gram.T @ df @ df.T
            #       4x3   3x4   4x4     4x4     4x4     4x4     4x3   3x4

    for k,v in M.items():
        M[k] = M[k] / (b + 1)

    remove_hooks(model)
    return M


def transform_network(model, dataloader, u, s, v):
    """ Calculate the new_edges of the model using the specified method """
    edges = {}
    model.derivatives = []
    add_hooks(model, [model.compute_derivatives_hook, model.grab_activations_hook])

    for b, (x, label) in enumerate(dataloader):
        if b == 10:
            break
        model.derivatives = []
        model(x.reshape(len(x), -1).to(device))

        for l in range(len(model.layers)):
            # a = torch.diag(s[l + 1])
            # b = u[l + 1]
            # c = model.derivatives[l].T
            # d = u[l].T
            # e = torch.pow(torch.diag(s[l]), -1)

            s_d = torch.diag(s[l])
            s_invrse = torch.where(s_d != 0, torch.pow(s_d, -1), torch.tensor([1.0], device=s[l].device))

            a = torch.diag(s[l+1]).shape
            b = u[l+1].shape
            c = model.derivatives[l].T.shape
            d = u[l].T.shape
            e = s_invrse.shape

            ls = torch.diag(s[l+1]) @ (u[l+1] @ model.activations[l+1].T)
            m = torch.diag(s[l+1]) @ (u[l+1] @ (model.derivatives[l].T @ (u[l].T @ s_invrse))) # TODO bug here since adding bias
            rs = torch.diag(s[l]) @ (u[l] @ model.activations[l].T)
            edges.setdefault(l, 0)
            value = ls * m * rs.T
            edges[l] += value

    for k,v in edges.items():
        edges[k] = edges[k] / (b + 1)

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
    main(data)






