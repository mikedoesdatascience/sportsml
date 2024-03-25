from typing import List

import dgl
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl

from .features import GRAPH_FEATURES, FEATURE_COLUMNS
from ...utils.datamodule import HeteroGraphDataModule


class NBAHeteroGraphDataModule(HeteroGraphDataModule):
    num_feats: int = len(GRAPH_FEATURES)
    num_targets: int = 1

    def __init__(
        self,
        games: pd.DataFrame,
        batch_size: int = 4,
        split_type: str = "random",
        splits: List[int] = [0.8, 0.1, 0.1],
        num_workers: int = 4,
    ):
        super().__init__(
            games=games,
            feature_columns=GRAPH_FEATURES,
            target_columns=["y"],
            win_column="won",
            loc_column="home",
            season_column="SEASON",
            date_column="GAME_DATE",
            team_column="TEAM_ABBREVIATION",
            num_nodes=30,
            batch_size=batch_size,
            split_type=split_type,
            splits=splits,
            num_workers=num_workers,
        )


class NBAGameDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df=None,
        val_df=None,
        test_df=None,
        feature_columns=FEATURE_COLUMNS,
        target_columns=["PLUS_MINUS"],
        batch_size=64,
        splits=[0.8, 0.1, 0.1],
        num_workers=4,
    ):
        super().__init__()
        self.df = df
        self.val_df = val_df
        self.test_df = test_df
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.batch_size = batch_size
        self.splits = splits
        self.num_workers = num_workers

    def df_to_dataset(self, df):
        X = df[self.feature_columns].values
        y = df[self.target_columns].values
        return torch.utils.data.TensorDataset(
            torch.from_numpy(X).float(), torch.from_numpy(y).float()
        )

    def setup(self, stage="train"):
        if self.val_df is None and self.test_df is None:
            self.ds = self.df_to_dataset(self.df)
            (
                self.train_ds,
                self.val_ds,
                self.test_ds,
            ) = torch.utils.data.random_split(self.ds, self.splits)
        else:
            self.train_ds = self.df_to_dataset(self.df)
            self.val_ds = self.df_to_dataset(self.val_df)
            self.test_ds = self.df_to_dataset(self.test_df)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class NBAGraphDataset(object):
    def __init__(
        self,
        df,
        feature_columns=GRAPH_FEATURES,
        target_columns=["PLUS_MINUS"],
    ):
        self.df = df
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        seasons = df["SEASON"].unique()
        self.dates = []
        for season in seasons:
            s = df[df["SEASON"] == season]
            for date in s["GAME_DATE"].unique():
                if s[s["GAME_DATE"] < date]["TEAM_ABBREVIATION"].unique().size == 30:
                    self.dates.extend(
                        s[s["GAME_DATE"] >= date]["GAME_DATE"]
                        .str.replace("-", "")
                        .astype(int)
                        .unique()
                        .tolist()
                    )
                    break
        self.graph = self.generate_graph()

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]
        g = self.graph.edge_subgraph(
            (self.graph.edata["date"] < date),
            relabel_nodes=False,
        )
        g = g.edge_subgraph(
            g.edata["season"] == g.edata["season"].max(), relabel_nodes=False
        )
        g.edata["train"] = g.edata["date"] != g.edata["date"].max()
        g.edata["w"] = (1 / (g.edata["date"].max() + 1 - g.edata["date"])).reshape(
            -1, 1
        )
        return g

    def generate_graph(self):
        g = dgl.graph(
            (
                torch.from_numpy(self.df["src"].values),
                torch.from_numpy(self.df["target"].values),
            ),
            num_nodes=30,
        )
        g.edata["f"] = torch.from_numpy(self.df[self.feature_columns].values).float()
        g.edata["y"] = torch.from_numpy(self.df[self.target_columns].values).float()
        g.edata["p"] = torch.from_numpy(self.df[["HOME"]].values).float()
        g.edata["rest"] = torch.from_numpy(self.df[["REST"]].values).float()
        g.edata["date"] = torch.from_numpy(
            self.df["GAME_DATE"].str.replace("-", "").values.astype(int)
        )
        g.edata["season"] = torch.from_numpy(self.df["SEASON"].values.astype(int))
        return g


class NBAGraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df,
        val_df=None,
        test_df=None,
        feature_columns=GRAPH_FEATURES,
        target_columns=["PLUS_MINUS"],
        batch_size=64,
        split_type="random",
        splits=[0.8, 0.1, 0.1],
        num_workers=4,
    ):
        super().__init__()
        self.df = df
        self.val_df = val_df
        self.test_df = test_df
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.batch_size = batch_size
        self.split_type = split_type
        self.splits = splits
        self.num_workers = num_workers

    def get_latest_date(self):
        max_train_date = self.df.iloc[self.train_ds.indices]["GAME_DATE"].max()
        max_val_date = self.df.iloc[self.val_ds.indices]["GAME_DATE"].max()
        return max(max_train_date, max_val_date)

    def setup(self, stage="train"):
        self.ds = NBAGraphDataset(self.df, self.feature_columns, self.target_columns)
        if self.split_type == "random":
            (
                self.train_ds,
                self.val_ds,
                self.test_ds,
            ) = torch.utils.data.random_split(self.ds, self.splits)
        elif self.split_type == "time":
            idx = (len(self.ds) * np.array(self.splits).cumsum()).astype(int)[:2]
            train_idx, val_idx, test_idx = np.array_split(np.arange(len(self.ds)), idx)
            self.train_ds = torch.utils.data.Subset(self.ds, train_idx.tolist())
            self.val_ds = torch.utils.data.Subset(self.ds, val_idx.tolist())
            self.test_ds = torch.utils.data.Subset(self.ds, test_idx.tolist())
        elif self.split_type is None:
            self.train_ds = self.ds
        else:
            raise ValueError(f"split type {self.split_type} not supported")
        if self.val_df is not None:
            self.val_ds = NBAGraphDataset(
                self.val_df, self.feature_columns, self.target_columns
            )
        if self.test_df is not None:
            self.test_ds = NBAGraphDataset(
                self.test_df, self.feature_columns, self.target_columns
            )

    def train_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return dgl.dataloading.GraphDataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
