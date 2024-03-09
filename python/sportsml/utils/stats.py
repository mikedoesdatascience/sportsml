from typing import List

import pandas as pd


def process_averages(
    games: pd.DataFrame,
    stats_columns: List[str],
    sort_columns: List[str],
    off_groupby_columns: List[str],
    def_groupby_columns: List[str],
    rolling_windows: List[int],
) -> pd.DataFrame:
    games = games.sort_values(sort_columns)
    avg = games.copy().drop(stats_columns, axis=1)
    off_avg_stats = (
        games.groupby(off_groupby_columns)[stats_columns]
        .expanding()
        .mean()
        .groupby(off_groupby_columns)
        .shift(1)
        .droplevel([0, 1])
    )
    off_avg_stats.columns = [f"off_{col}_avg" for col in off_avg_stats.columns]
    avg = avg.merge(off_avg_stats, left_index=True, right_index=True)
    def_avg_stats = (
        games.groupby(def_groupby_columns)[stats_columns]
        .expanding()
        .mean()
        .groupby(def_groupby_columns)
        .shift(1)
        .droplevel([0, 1])
    )
    def_avg_stats.columns = [f"def_{col}_avg" for col in def_avg_stats.columns]
    avg = avg.merge(def_avg_stats, left_index=True, right_index=True)
    for rolling_window in rolling_windows:
        off_rolling_stats = (
            games.groupby(off_groupby_columns)[stats_columns]
            .rolling(rolling_window, 1, closed="left")
            .mean()
            .droplevel([0, 1])
        )
        off_rolling_stats.columns = [
            f"off_{col}_rolling_{rolling_window}" for col in off_rolling_stats.columns
        ]
        avg = avg.merge(off_rolling_stats, left_index=True, right_index=True)
        def_rolling_stats = (
            games.groupby(def_groupby_columns)[stats_columns]
            .rolling(rolling_window, 1, closed="left")
            .mean()
            .droplevel([0, 1])
        )
        def_rolling_stats.columns = [
            f"def_{col}_rolling_{rolling_window}" for col in def_rolling_stats.columns
        ]
        avg = avg.merge(def_rolling_stats, left_index=True, right_index=True)
    return avg
