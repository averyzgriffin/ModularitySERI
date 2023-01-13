import os
import torch

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
                # print("Loss: ", self.loss)
                self.loss.backward()
                self.opt.step()

        print("Final Loss: ", self.loss)


class TrainerRetina(Trainer):

    def __init__(self, model, loss_fc, epochs, dataloader, device):
        super(TrainerRetina, self).__init__(model, loss_fc, epochs, dataloader, device)
        self.goal_and = 1

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
                # print("Loss: ", self.loss)
                self.loss.backward()
                self.opt.step()

        print("Final Loss: ", self.loss)

    @staticmethod
    def compute_loss(model, loss_fn, dataloader, device, goal_and):
        for b, (x, label) in enumerate(dataloader):
            x = x.to(device)
            _and, _or = label
            result = _and if goal_and else _or
            prediction = model(x)
            loss = loss_fn(prediction.view(-1), result.float().to(device))
            print("Loss: ", loss)

