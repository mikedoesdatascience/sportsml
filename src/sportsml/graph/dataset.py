import pandas as pd
from torch_geometric.data import Dataset

from .graph import create_graph


class GraphDataset(Dataset):
    def __init__(
        self,
        games: pd.DataFrame,
        stats_columns: list[str],
        target_column: str,
        season_column: str,
        date_column: str,
    ):
        self.graph = create_graph(
            games=games,
            stats_columns=stats_columns,
            target_column=target_column,
            season_column=season_column,
            date_column=date_column,
        )
        self.stats_columns = stats_columns
        self.target_column = target_column
        self.season_column = season_column
        self.date_column = date_column

        self.dates = self.filter_valid_dates(games.copy())

    def filter_valid_dates(self, games):
        games["gp"] = (
            games.sort_values([self.season_column, self.date_column])
            .groupby([self.season_column, "src"])
            .cumcount()
        )
        min_gp = games.groupby([self.season_column, self.date_column])["gp"].min() > 0
        return min_gp[min_gp].index.tolist()

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx: int):
        season, date = self.dates[idx]

        graph = self.graph.edge_subgraph(
            (self.graph.season == season) & (self.graph.date <= date)
        )

        return graph

    def get_latest_graph(self):
        return self[self.dates.index(sorted(self.dates)[-1])]