from typing import List

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .dataset import GraphDataset


class GraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        games: pd.DataFrame,
        stats_columns: List[str],
        target_column: str,
        season_column: str,
        date_column: str,
        train_seasons: list[int] = None,
        val_seasons: list[int] = None,
        test_seasons: list[int] = None,
        batch_size=8,
        num_workers=4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        unique_seasons = sorted(games[season_column].unique(), reverse=True)

        if val_seasons is None:
            val_seasons = unique_seasons[1:2]
        if train_seasons is None:
            train_seasons = unique_seasons[2:]

        if set(train_seasons) & set(val_seasons):
            raise ValueError("Train and validation seasons overlap")
        
        train_games = games[games[season_column].isin(train_seasons)]
        val_games = games[games[season_column].isin(val_seasons)]

        self.train_ds = GraphDataset(
            games=train_games,
            stats_columns=stats_columns,
            target_column=target_column,
            season_column=season_column,
            date_column=date_column,
        )
        self.val_ds = GraphDataset(
            games=val_games,
            stats_columns=stats_columns,
            target_column=target_column,
            season_column=season_column,
            date_column=date_column,
        )

        if test_seasons is None:
            test_seasons = unique_seasons[:1]

        if test_seasons:
            if set(train_seasons) & set(test_seasons):
                raise ValueError("Train and test seasons overlap")
            if set(val_seasons) & set(test_seasons):
                raise ValueError("Validation and test seasons overlap")

            test_games = games[games[season_column].isin(test_seasons)]

            self.test_ds = GraphDataset(
                games=test_games,
                stats_columns=stats_columns,
                target_column=target_column,
                season_column=season_column,
                date_column=date_column,
            )
        else:
            self.test_ds = None

    def setup(self, stage: str = "train"):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )