from engine.predictor import Predictor
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer(Predictor):
    def __init__(self, model: nn.Module, trainloader: DataLoader, hyperparameters):
        super().__init__(model, trainloader, hyperparameters)
        self.model = model
        
        # hyperparameters
        self.loss_fn = hyperparameters.loss_fn
        self.optimizer = hyperparameters.optimizer

    def _step(self, x, y):
        '''
        Trains a batch input x.
        Returns a batch prediction and losses
        '''
        self.model.train()
        predicted = self.model.forward(x)
        loss = self.loss_fn(predicted, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()