import numpy as np
import torch
import torch.nn as nn


def add_layers(model, dimensions):
    layers = []
    for i, (dimension) in enumerate(dimensions):
        fc = nn.Linear(dimension, dimensions[i + 1])
        setattr(model, f"fc{i}", fc)
        layers.append(fc)
        if len(dimensions) == i + 2:
            break

    return layers


class OrthogMLP(nn.Module):
    layers = []

    def __init__(self, *dimensions):
        super(OrthogMLP, self).__init__()

        self.layers = add_layers(self, dimensions)
        self.relu = nn.LeakyReLU()
        self.activations = []

    def forward(self, x):
        x_and_bias = torch.cat((x,torch.ones(x.shape[0],1)), dim=1)
        self.activations = [x_and_bias]
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            x_and_bias = torch.cat((x,torch.ones(x.shape[0],1)), dim=1)
            self.activations.append(x_and_bias)

        # self.collect_activations(activation)  # TODO taking this out for orthog decomposition

        return self.layers[-1](x)

    def get_loss(self, dataloader, loss_fc, device):
        losses = []
        for b, (x, y) in enumerate(dataloader):
            x = x.to(device)
            prediction = self(x.reshape(len(x), -1))
            loss = loss_fc(prediction, y.to(device))
            losses.append(loss.item())
        return np.mean(losses)

    def get_x_y_batches(self, dataloader, device):
        all_batches_x = []
        all_batches_y = []
        for b, (x, y) in enumerate(dataloader):
            x = x.reshape(len(x), -1).to(device)
            prediction = self(x)
            all_batches_x.append(x)
            all_batches_y.append(prediction)

        return torch.cat(all_batches_x, dim=0), torch.cat(all_batches_y, dim=0)


class SimpleMLP(nn.Module):
    layers = []

    def __init__(self, *dimensions):
        super(SimpleMLP, self).__init__()

        self.layers = add_layers(self, dimensions)

        self.relu = nn.LeakyReLU()

        self.activations_dict = dict()
        self.clusters = []
        self.m1_patterns, self.m2_patterns = [], []
        self.m1_counter = dict()
        self.m2_counter = dict()
        self.joint_counter = dict()

        self.avery = 0

    def forward(self, x):
        activation = [x]
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            # x = (x>0.5).float()
            # activation.append((x > 0.3).float())# TODO taking this out for orthog decomposition

        # activation.append( (self.layers[-1](x) > 0.5).float() )# TODO taking this out for orthog decomposition
        # self.collect_activations(activation)  # TODO taking this out for orthog decomposition

        return self.layers[-1](x)

    def collect_activations(self, activation):
        flattened_activation = []
        for layer in activation[:-1]:
            flattened_activation += layer.squeeze().tolist()
        flattened_activation += activation[-1].tolist()[0]

        self.activations_dict[tuple(flattened_activation[:4])] = flattened_activation
        self.extract_modules(self.clusters, flattened_activation)

    def extract_modules(self, clusters, activation):
        m1_modules = []
        m2_modules = []

        # m1 = [x for c, x in zip(clusters, activation) if c == 0]
        # m2 = [x for c, x in zip(clusters, activation) if c == 1]
        # m1_modules.append(m1)
        # m2_modules.append(m2)

        m1 = activation[:4]
        m2 = [activation[-1]]
        m1_modules.append(m1)
        m2_modules.append(m2)

        self.update_dict_counter(tuple(m1), self.m1_counter)
        self.update_dict_counter(tuple(m2), self.m2_counter)
        self.update_dict_counter(tuple([tuple(m1),tuple(m2)]), self.joint_counter)

        # return self.get_unique(m1_modules), self.get_unique(m2_modules)

    @staticmethod
    def get_unique(lst):
        return [list(a) for a in set(tuple(b) for b in lst)]

    @staticmethod
    def update_dict_counter(key, d: dict):
        if key not in d:
            d[key] = 1
        else:
            d[key] += 1

