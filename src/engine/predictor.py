import torch.nn as nn
from torch.utils.data import DataLoader

class Predictor():
    def __init__(self, model: nn.Module, dataloader: DataLoader, hyperparameters):
        self.model = model
        self.dataloader = dataloader
    
    def _step(self):
        raise NotImplementedError()
    
    def get_avg_loss_for_epoch(self):
        running_loss = 0
        for idx, (features, labels) in enumerate(self.dataloader):
            loss = self._step(features, labels)
            running_loss += loss
        
        return running_loss / len(self.dataloader.dataset)
    
    def run_epoch(self):
        return self.get_avg_loss_for_epoch()
    