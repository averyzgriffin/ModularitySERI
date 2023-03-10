import os
import torch

from eigen import compute_eigens
from gram import compute_grams, preprocess_lams


conf_path = os.getcwd()
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Trainer:

    def __init__(self, model, N, loss_fc, lr, opt, regularization, epochs, dataloader, test_loader, device, save_path, model_name):
        self.model = model
        self.N = N
        self.epochs = epochs
        self.dataloader = dataloader
        self.test_loader = test_loader
        self.device = device
        self.loss_fc = loss_fc
        self.lr = lr
        self.regularization = regularization
        self.opt = opt(lr=self.lr, params=model.parameters(), weight_decay=self.regularization)
        self.compute_validation = False
        self.loss = None
        self.valid_loss = []
        self.gram_lams = []
        self.save_path = save_path
        self.model_name = model_name

    def train(self):
        for i in range(self.epochs):
            print("Epoch ", i)
            epoch_loss = 0
            for b, (x, y) in enumerate(self.dataloader):
                x = x.to(self.device)
                prediction = self.model(x.reshape(len(x), -1))
                self.opt.zero_grad()
                self.loss = self.loss_fc(prediction, y.to(self.device))
                self.loss.backward()
                self.opt.step()
                epoch_loss += self.loss

            torch.save(self.model.state_dict(), f'{self.save_path}/{self.model_name}_epoch{str(i).zfill(3)}.pt')
            # print(f"Validation Accuracy: ", self.validate())
            # self.valid_loss.append(self.validate())
            # self.gram_lams.append(self.compute_grams())

            print(f"Epoch {i} Average Loss: ", epoch_loss / (b+1))
        print("Final Epoch Loss: ", epoch_loss / (b+1))

    def validate(self):
        # score = self.model.get_loss(self.test_loader, self.loss_fc, self.device) / len(self.test_loader)
        score = self.model.get_accuracy(self.test_loader, self.device)
        return score

    def compute_grams(self):
        print("Gram Computations")
        grams = compute_grams(self.model, self.dataloader, True, self.device)
        U, lam = compute_eigens(grams)
        lam = preprocess_lams(lam, self.N[2:])
        return lam


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

