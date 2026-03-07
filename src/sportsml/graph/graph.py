import pandas as pd
import torch
import torch_geometric as pyg


def create_graph(
    games: pd.DataFrame,
    stats_columns: list[str],
    target_column: str,
    season_column: str = "season",
    date_column: str = "date",
):
    num_nodes = games["src"].max() + 1
    edge_index = torch.tensor([games["src"].tolist(), games["dst"].tolist()])
    y = torch.tensor(games[[target_column]].values, dtype=torch.float)
    edge_attr = torch.tensor(games[stats_columns].values, dtype=torch.float)
    season = torch.tensor(games[season_column].values, dtype=torch.float)
    date = torch.tensor(games[date_column].values, dtype=torch.float)

    graph = pyg.data.Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
        y=y,
        edge_attr=edge_attr,
        season=season,
        date=date,
    )

    return graph
