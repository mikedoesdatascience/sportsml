import pandas as pd
import torch
import torch_geometric as pyg
from torch_geometric.data import Data


def create_graph(
    games: pd.DataFrame,
    stats_columns: list[str],
    target_column: str,
    train_mask: list[bool] = None,
):
    num_nodes = games["src"].max() + 1
    edge_index = torch.tensor([games["src"].tolist(), games["dst"].tolist()])
    y = torch.tensor(games[[target_column]].values, dtype=torch.float)
    edge_attr = torch.tensor(games[stats_columns].values, dtype=torch.float)

    train_mask = (
        torch.tensor(train_mask, dtype=torch.bool)
        if train_mask is not None
        else torch.ones(len(games), dtype=torch.bool)
    )

    graph = Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
        y=y,
        edge_attr=edge_attr,
        train_mask=train_mask,
    )

    return graph
