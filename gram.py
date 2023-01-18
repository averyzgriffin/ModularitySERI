import torch


def compute_grams(model, dataloader, per_layer, device):
    for b, (x, label) in enumerate(dataloader):
        prediction = model(x.reshape(len(x), -1).to(device))
        activations = model.activations
        if b == 0:
            Grams = [torch.matmul(act.transpose(0, 1), act) for act in activations]
        else:
            for l in range(len(activations)):
                Grams[l] += torch.matmul(activations[l].transpose(0, 1), activations[l])

    Grams = [gram / (b+1) for gram in Grams]
    return Grams


def preprocess_lams(lams: list, N: list):
    repeated_tensors = [lams[i].repeat(N[i]) for i in range(len(lams))]
    return torch.cat(repeated_tensors, dim=0).detach()


def preprocess_lams_full_network(lams: list, N: list):
    x = [lams[0][N[i]:N[i+1]+1].repeat(N[i]) for i in range(len(N))]
    return torch.cat(x, dim=0).detach()


def repeat_and_concatenate(lam, N):
    ind = [9, 9, 5, 3]
    split_tensors = []
    start_idx = 0
    for i in ind:
        end_idx = start_idx + i
        split_tensors.append(lam[start_idx:end_idx])
        start_idx = end_idx
    repeated_tensors = [split_tensors[j].repeat(N[j]) for j in range(len(N))]
    concatenated_tensor = torch.cat(repeated_tensors)
    return concatenated_tensor.detach()
