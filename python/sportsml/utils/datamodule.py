from typing import List

import dgl
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch

from .dataset import HeteroGraphDataset


class HeteroGraphDataModule(pl.LightningDataModule):
    num_feats: int = None
    num_target: int = None

    def __init__(
        self,
        games: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        win_column: str,
        loc_column: str,
        season_column: str,
        date_column: str,
        team_column: str,
        num_nodes: int,
        batch_size=8,
        split_type="random",
        splits=[0.8, 0.1, 0.1],
        num_workers=4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_feats = len(feature_columns)
        self.num_targets = len(target_columns)

        self.ds = HeteroGraphDataset(
            games=games,
            feature_columns=feature_columns,
            target_columns=target_columns,
            win_column=win_column,
            loc_column=loc_column,
            season_column=season_column,
            date_column=date_column,
            team_column=team_column,
            num_nodes=num_nodes,
        )
        ds_idx = np.arange(len(self.ds))
        val_idx = int(splits[0] * len(self.ds))
        test_idx = int((splits[0] + splits[1]) * len(self.ds))
        np.random.shuffle(ds_idx)
        self.train_idx = ds_idx[:val_idx]
        self.val_idx = ds_idx[val_idx:test_idx]
        self.test_idx = ds_idx[test_idx:]

    def setup(self, stage: str = "train"):
        pass

    def train_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
            self.ds,
            sampler=torch.utils.data.SubsetRandomSampler(self.train_idx),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
            self.ds,
            sampler=torch.utils.data.SubsetRandomSampler(self.val_idx),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
            self.ds,
            sampler=torch.utils.data.SubsetRandomSampler(self.test_idx),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
