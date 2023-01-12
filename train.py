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


class Trainer:

    def __init__(self, model, loss_fc, epochs, dataloader, device):
        self.model = model
        self.loss_fc = loss_fc
        self.epochs = epochs
        self.dataloader = dataloader
        self.device = device
        self.loss = None
        self.opt = torch.optim.Adam(lr=1e-3, params=model.parameters())

    def train(self):
        for i in range(self.epochs):
            for b, (x, y) in enumerate(self.dataloader):
                x = x.to(self.device)
                prediction = self.model(x)
                self.opt.zero_grad()
                self.loss = self.loss_fc(prediction.view(-1), y.float().to(self.device))
                print("Loss: ", self.loss)
                self.loss.backward()
                self.opt.step()

        print("Final Loss: ", self.loss)


class TrainerRetina(Trainer):

    def __init__(self, model, loss_fc, epochs, dataloader, device):
        super(TrainerRetina, self).__init__(model, loss_fc, epochs, dataloader, device)

    def train(self):
        goal_and = 1
        for i in range(self.epochs):
            for b, (x, label) in enumerate(self.dataloader):
                x = x.to(self.device)
                _and, _or = label
                result = _and if goal_and else _or
                prediction = self.model(x)
                self.opt.zero_grad()
                self.loss = self.loss_fc(prediction.view(-1), result.float().to(self.device))
                print("Loss: ", self.loss)
                self.loss.backward()
                self.opt.step()

        print("Final Loss: ", self.loss)
