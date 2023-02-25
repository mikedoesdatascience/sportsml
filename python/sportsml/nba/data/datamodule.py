import torch
import pytorch_lightning as pl

from .utils import get_training_data


class NBAGameDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
    
    def setup(self, stage):
        X, y = get_training_data()
        self.ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X).float(), torch.from_numpy(y.reshape(-1, 1)).float()
        )
        self.train_ds, self.val_ds, self.test_ds = torch.utils.data.random_split(
            self.ds,
            [0.8, 0.1, 0.1]
        )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)