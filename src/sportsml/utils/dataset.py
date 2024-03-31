from typing import List

import pandas as pd

from .heterograph import heterograph_encoder, heterograph_predictor


class HeteroGraphDataset:
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
    ):
        self.games = games.copy()
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.win_column = win_column
        self.loc_column = loc_column
        self.season_column = season_column
        self.date_column = date_column
        self.team_column = team_column
        self.num_nodes = num_nodes

        self.dates = self.filter_valid_dates()

    def filter_valid_dates(self):
        self.games["gp"] = self.games.groupby(
            [self.season_column, self.team_column]
        ).cumcount()
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
            & (self.games[self.date_column] < date)
        ]
        encoder_graph = heterograph_encoder(
            games["src"].values,
            games["dst"].values,
            feat=games[self.feature_columns].values,
            win_mask=games[self.win_column].values,
            num_nodes=self.num_nodes,
        )

        pgames = self.games[
            (self.games[self.season_column] == season)
            & (self.games[self.date_column] == date)
        ]
        predictor_graph = heterograph_predictor(
            pgames["src"].values,
            pgames["dst"].values,
            feat=pgames[self.feature_columns].values,
            loc=pgames[self.loc_column].values,
            y=pgames[self.target_columns].values,
            num_nodes=self.num_nodes,
        )

        return encoder_graph, predictor_graph
