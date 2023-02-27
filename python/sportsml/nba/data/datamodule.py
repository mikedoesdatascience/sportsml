import torch
import pytorch_lightning as pl

from .features import FEATURE_COLUMNS
from .utils import get_training_data


class NBAGameDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df=None,
        val_df=None,
        test_df=None,
        feature_columns=FEATURE_COLUMNS,
        target_column='PLUS_MINUS',
        batch_size=64,
        splits=[0.8, 0.1, 0.1]
    ):
        super().__init__()
        if df is None:
            df = get_training_data()
        self.df = df
        self.val_df = val_df
        self.test_df = test_df
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.batch_size = batch_size
        self.splits = splits

    def df_to_dataset(self, df):
        X = df[self.feature_columns].values
        y = df[[self.target_column]].values
        return torch.utils.data.TensorDataset(
            torch.from_numpy(X).float(), torch.from_numpy(y).float()
        )
    
    def setup(self, stage):
        if self.val_df is None and self.test_df is None:
            self.ds = self.df_to_dataset(self.df)
            self.train_ds, self.val_ds, self.test_ds = torch.utils.data.random_split(
                self.ds,
                self.splits
            )
        else:
            self.train_ds = self.df_to_dataset(self.df)
            self.val_ds = self.df_to_dataset(self.val_df)
            self.test_ds = self.df_to_dataset(self.test_df)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)