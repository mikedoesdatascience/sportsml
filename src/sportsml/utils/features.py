import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

from .stats import process_averages


def process_features(
    games: pd.DataFrame,
    stats_columns: list[str],
    meta_columns: list[str],
    categorical_columns: list[str],
    game_id_column: str = "game_id",
    season_column: str = "season",
    date_column: str = "date",
    team_column: str = "team",
    rolling_windows: list[int] = None,
    use_all_data: bool = False,
):
    X_cat = OneHotEncoder(
        sparse_output=False, min_frequency=10, handle_unknown="ignore"
    ).fit_transform(games[categorical_columns].values)
    X_meta = KNNImputer(n_neighbors=10).fit_transform(games[meta_columns].values)
    X_avgs = process_averages(
        games=games,
        stats_columns=stats_columns,
        game_id_column=game_id_column,
        season_column=season_column,
        date_column=date_column,
        team_column=team_column,
        rolling_windows=rolling_windows,
        use_all_data=use_all_data,
    )
    X = np.hstack([X_avgs, X_meta, X_cat])
    return X
