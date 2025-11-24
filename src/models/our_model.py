import torch.nn as nn
from data_utils.config import ModelConfig

class CaseEmbeddingsModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def save(self, save_dir: str):
        pass

    def load_checkpoint(self, checkpoint_path: str):
        pass