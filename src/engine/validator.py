from engine.predictor import Predictor
import torch.nn as nn
from torch.utils.data import DataLoader

class Validator(Predictor):
    def __init__(self, model: nn.Module, validateloader: DataLoader, hyperparameters):
        super().__init__(model, validateloader, hyperparameters)
        self.model = model
        self.loss_fn = hyperparameters.loss_fn
    
    def _step(self, x, y):
        '''
        Tests a batch input
        Returns an inference and losses
        '''
        self.model.eval()
        predicted = self.model.forward(x)
        loss = self.loss_fn(predicted, y)
        return loss.item()