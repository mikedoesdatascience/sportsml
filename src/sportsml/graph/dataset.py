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
        self.games = games.copy()
        self.stats_columns = stats_columns
        self.target_column = target_column
        self.season_column = season_column
        self.date_column = date_column

        self.dates = self.filter_valid_dates()

    def filter_valid_dates(self):
        self.games["gp"] = (
            self.games.sort_values([self.season_column, self.date_column])
            .groupby([self.season_column, "src"])
            .cumcount()
        )
        min_gp = (
            self.games.groupby([self.season_column, self.date_column])["gp"].min() > 0
        )
        return min_gp[min_gp].index.tolist()

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx: int):
        season, date = self.dates[idx]

        games = self.games[
            (self.games[self.season_column] == season)
            & (self.games[self.date_column] <= date)
        ]

        train_mask = (games[self.date_column] < date).tolist()

        graph = create_graph(
            games=games,
            stats_columns=self.stats_columns,
            target_column=self.target_column,
            train_mask=train_mask
        )

        return graph
