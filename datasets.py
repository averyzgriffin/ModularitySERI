import itertools
import numpy as np
import random
import torch
from torch.utils.data import Dataset


class ModularArithmeticDataset(torch.utils.data.Dataset):
    """
    Dataset used in Neel Nanda's modular arithmetic work.
    """
    def __init__(self, p, fn_name, device, split=0.8, seed=0, train=True):
        self.p = p
        self.fn_name = fn_name
        self.device = device
        self.split = split
        self.seed = seed
        self.train = train

        self.fns_dict = {'add': lambda x, y: (x + y) % self.p, 'subtract': lambda x, y: (x - y) % self.p,
                         'x2xyy2': lambda x, y: (x ** 2 + x * y + y ** 2) % self.p}
        self.fn = self.fns_dict[fn_name]
        self.x, self.labels = self.construct_dataset()
        self.train_x, self.train_labels, self.test_x, self.test_labels = self.split_dataset()

    def __getitem__(self, index):
        if self.train:
            return self.train_x[index], self.train_labels[index]
        else:
            return self.test_x[index], self.test_labels[index]

    def __len__(self):
        if self.train:
            return len(self.train_x)
        else:
            return len(self.test_x)

    def construct_dataset(self):
        x = torch.tensor([(i, j, self.p) for i in range(self.p) for j in range(self.p)]).to(self.device)
        y = torch.tensor([self.fn(i, j) for i, j, _ in x]).to(self.device)
        return x, y

    def split_dataset(self):
        random.seed(self.seed)
        indices = list(range(len(self.x)))
        random.shuffle(indices)
        div = int(self.split * len(indices))
        train_indices, test_indices = indices[:div], indices[div:]
        train_x, train_labels = self.x[train_indices], self.labels[train_indices]
        test_x, test_labels = self.x[test_indices], self.labels[test_indices]
        return train_x, train_labels, test_x, test_labels


class RetinaDataset(Dataset):
    """
    This dataset provides a random one-hot encoded sample
    from the 4x2 retina grid problem as one tensor.
    """

    def __init__(self, size):
        super(RetinaDataset, self).__init__()
        self.samples = self.generate_samples(size)
        self.length = len(self.samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.samples[idx]["pixels"]
        data = torch.Tensor(x)
        and_label = self.samples[idx]["and_label"]
        or_label = self.samples[idx]["or_label"]
        return data, (and_label, or_label)

    def generate_samples(self, size):
        return [dict({"pixels": np.array(seq), "and_label": self.and_label_sample(seq, size),
                      "or_label": self.or_label_sample(seq, size)}) for seq in itertools.product([0,1], repeat=size)]

    @staticmethod
    def and_label_sample(sample, size):
        left = False
        right = False

        # if (sum(sample[0:4]) >= 3) or (sum(sample[0:2]) >= 1 and sum(sample[2:4]) == 0): left = True
        # if (sum(sample[4:8]) >= 3) or (sum(sample[6:8]) >= 1 and sum(sample[4:6]) == 0): right = True

        if (sum(sample[0:2]) >= 2) or (sum(sample[0:1]) >= 1 and sum(sample[1:2]) == 0): left = True
        if (sum(sample[2:4]) >= 2) or (sum(sample[3:4]) >= 1 and sum(sample[2:3]) == 0): right = True

        # if (sum(sample[0:(size // 2)]) >= int(size / 2 * .75)) or (sum(sample[0:(size // 4)]) >= 1 and sum(sample[(size // 4):(size // 2)]) == 0): left = True
        # if (sum(sample[(size // 2):size]) >= int(size / 2 * .75)) or (sum(sample[int(size * .75):size]) >= 1 and sum(sample[(size // 2):int(size * .75)]) == 0): right = True

        if left and right: return 1
        elif left: return 0
        elif right: return 0
        else: return 0

    @staticmethod
    def or_label_sample(sample, size):
        left = False
        right = False

        # if (sum(sample[0:4]) >= 3) or (sum(sample[0:2]) >= 1 and sum(sample[2:4]) == 0): left = True
        # if (sum(sample[4:8]) >= 3) or (sum(sample[6:8]) >= 1 and sum(sample[4:6]) == 0): right = True
        if (sum(sample[0:2]) >= 2) or (sum(sample[0:1]) >= 1 and sum(sample[1:2]) == 0): left = True
        if (sum(sample[2:4]) >= 2) or (sum(sample[3:4]) >= 1 and sum(sample[2:3]) == 0): right = True
        # if (sum(sample[0:(size // 2)]) >= int(size / 2 * .75)) or (sum(sample[0:(size // 4)]) >= 1 and sum(sample[(size // 4):(size // 2)]) == 0): left = True
        # if (sum(sample[(size // 2):size]) >= int(size / 2 * .75)) or (sum(sample[int(size * .75):size]) >= 1 and sum(sample[(size // 2):int(size * .75)]) == 0): right = True

        if left and right: return 1
        elif left: return 1
        elif right: return 1
        else: return 0


