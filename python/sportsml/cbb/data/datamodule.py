import dgl
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl

from .features import GRAPH_FEATURES


class CBBGraphDataset(object):
    def __init__(self, df, feature_columns=GRAPH_FEATURES, target_column="PlusMinus"):
        self.df = df
        self.feature_columns = feature_columns
        self.target_column = target_column
        seasons = df["Season"].unique()
        self.dates = []
        for season in seasons:
            s = df[df["Season"] == season]
            for date in s["DayNum"].unique():
                if (
                    s[s["DayNum"] < date]["TeamID"].unique().size
                    == s["TeamID"].unique().size
                ):
                    self.dates.extend(
                        [
                            [season, d]
                            for d in s[s["DayNum"] >= date]["DayNum"]
                            .astype(int)
                            .unique()
                            .tolist()
                        ]
                    )
                    break
        self.graph = self.generate_graph()

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        season, date = self.dates[idx]
        g = self.graph.edge_subgraph(
            (self.graph.edata["season"] == season) & (self.graph.edata["date"] <= date),
            relabel_nodes=False,
        )
        g.edata["train"] = g.edata["date"] != g.edata["date"].max()
        g.edata["w"] = (1 / (g.edata["date"].max() + 1 - g.edata["date"])).reshape(
            -1, 1
        )
        return g

    def generate_graph(self):
        g = dgl.graph(
            (
                torch.from_numpy(self.df["TeamID_OPP"].values),
                torch.from_numpy(self.df["TeamID"].values),
            ),
            num_nodes=self.df["TeamID"].max() + 1,
        )
        g.edata["f"] = torch.from_numpy(self.df[self.feature_columns].values).float()
        g.edata["y"] = torch.from_numpy(self.df[[self.target_column]].values).float()
        g.edata["p"] = torch.from_numpy(self.df[["Loc"]].values).float()
        g.edata["date"] = torch.from_numpy(self.df["DayNum"].values.astype(int))
        g.edata["season"] = torch.from_numpy(self.df["Season"].values.astype(int))
        return g


class CBBGraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df,
        feature_columns=GRAPH_FEATURES,
        target_column="PlusMinus",
        batch_size=64,
        split_type="random",
        splits=[0.8, 0.1, 0.1],
        num_workers=4,
    ):
        super().__init__()
        self.df = df
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.batch_size = batch_size
        self.split_type = split_type
        self.splits = splits
        self.num_workers = num_workers

    def setup(self, stage="train"):
        self.ds = CBBGraphDataset(self.df, self.feature_columns, self.target_column)
        if self.split_type == "random":
            self.train_ds, self.val_ds, self.test_ds = torch.utils.data.random_split(
                self.ds, self.splits
            )
        elif self.split_type == "time":
            idx = (len(self.ds) * np.array(self.splits).cumsum()).astype(int)[:2]
            train_idx, val_idx, test_idx = np.array_split(np.arange(len(self.ds)), idx)
            self.train_ds = torch.utils.data.Subset(self.ds, train_idx.tolist())
            self.val_ds = torch.utils.data.Subset(self.ds, val_idx.tolist())
            self.test_ds = torch.utils.data.Subset(self.ds, test_idx.tolist())
        else:
            raise ValueError(f"split type {self.split_type} not supported")

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
