from config import DataConfig
from torch.utils.data import Dataset, DataLoader

class CaseEmbeddingsData(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    

def get_dataloader(dataset: Dataset, config: DataConfig):
    return DataLoader(dataset, batch_size = config.batch_size, shuffle = config.shuffle)
