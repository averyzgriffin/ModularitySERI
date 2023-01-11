import itertools

import numpy as np
import torch
from torch.utils.data import Dataset


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


