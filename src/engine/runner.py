from data_utils.config import ExperimentConfig
from engine.predictor import Predictor
from engine.trainer import Trainer
from engine.validator import Validator
import torch.nn as nn
from torch.utils.data import DataLoader

# main class for kicking off training and validation 
class Runner():
    def __init__(self, model: nn.Module, trainloader: DataLoader, validateloader: DataLoader, config: ExperimentConfig):
        self.trainer = Trainer(model, trainloader, config)
        self.validator = Validator(model, validateloader, config)
        self.num_epochs = config.num_epochs
        self.train_losses, self.validation_losses = [], [] # can plot
    
    def _run_train_one_epoch(self):
        train_loss = self.trainer.run_epoch()
        self.train_losses.append(train_loss)

    def _run_eval_one_epoch(self):
        val_loss = self.validator.run_epoch()
        self.validation_losses.append(val_loss)

    def run_all(self):
        for epoch in range(self.num_epochs):
            self._run_train_one_epoch()
            self._run_eval_one_epoch()
        return self.train_losses, self.validation_losses