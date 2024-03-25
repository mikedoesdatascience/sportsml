import dgl
import torch

from .features import GRAPH_FEATURES
from .nodes import team_abr_map
from ...utils.dataset import HeteroGraphDataset


class CFBHeteroGraphDataset(HeteroGraphDataset):
    def __init__(self, games):
        super().__init__(
            games=games,
            feature_columns=GRAPH_FEATURES,
            target_columns=["result"],
            win_column="won",
            loc_column="home",
            season_column="season",
            date_column="week",
            team_column="team",
            num_nodes=320,
        )


class CFBGraphDataset(object):
    def __init__(
        self,
        df,
        feature_columns=GRAPH_FEATURES,
        target_columns=["result"],
    ):
        self.df = df
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.dates = (
            df[df["week"] != 1][["season", "week"]].drop_duplicates().values.tolist()
        )
        self.graph = self.generate_graph()

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        season, week = self.dates[idx]
        g = self.graph.edge_subgraph(
            (
                (self.graph.edata["season"] == season)
                & (self.graph.edata["week"] < week)
            ),
            relabel_nodes=False,
        )
        g.edata["train"] = g.edata["week"] != g.edata["week"].max()
        g.edata["w"] = (1 / (g.edata["week"].max() + 1 - g.edata["week"])).reshape(
            -1, 1
        )
        return g

    def generate_graph(self):
        g = dgl.graph(
            (
                torch.from_numpy(self.df["src"].values),
                torch.from_numpy(self.df["target"].values),
            ),
            num_nodes=len(team_abr_map),
        )
        g.edata["f"] = torch.from_numpy(self.df[self.feature_columns].values).float()
        g.edata["y"] = torch.from_numpy(self.df[self.target_columns].values).float()
        g.edata["p"] = torch.from_numpy(self.df[["home"]].values).float()
        g.edata["week"] = torch.from_numpy(self.df["week"].values)
        g.edata["season"] = torch.from_numpy(self.df["season"].values)
        return g
