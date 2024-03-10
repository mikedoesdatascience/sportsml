from typing import List

import pandas as pd


def process_averages(
    games: pd.DataFrame,
    stats_columns: List[str],
    season_column: str,
    date_column: str,
    team_column: str,
    opp_column: str,
    rolling_windows: List[int],
) -> pd.DataFrame:
    games = games.sort_values([season_column, date_column])
    avg = games.copy().drop(stats_columns, axis=1)
    avg_stats = (
        games.groupby([season_column, team_column])[stats_columns]
        .expanding()
        .mean()
        .groupby([season_column, team_column])
        .shift(1)
        .droplevel([0, 1])
    )
    avg_stats.columns = [f"{col}_avg" for col in avg_stats.columns]
    avg = avg.merge(avg_stats, left_index=True, right_index=True)
    opp_avg_stats = (
        games.groupby([season_column, opp_column])[stats_columns]
        .expanding()
        .mean()
        .groupby([season_column, opp_column])
        .shift(1)
        .droplevel([0, 1])
    )
    opp_avg_stats.columns = [f"OPP_{col}_avg" for col in opp_avg_stats.columns]
    avg = avg.merge(opp_avg_stats, left_index=True, right_index=True)
    for rolling_window in rolling_windows:
        rolling_stats = (
            games.groupby([season_column, team_column])[stats_columns]
            .rolling(rolling_window, 1, closed="left")
            .mean()
            .droplevel([0, 1])
        )
        rolling_stats.columns = [
            f"{col}_rolling_{rolling_window}" for col in rolling_stats.columns
        ]
        avg = avg.merge(rolling_stats, left_index=True, right_index=True)
        opp_rolling_stats = (
            games.groupby([season_column, opp_column])[stats_columns]
            .rolling(rolling_window, 1, closed="left")
            .mean()
            .droplevel([0, 1])
        )
        opp_rolling_stats.columns = [
            f"OPP_{col}_rolling_{rolling_window}" for col in opp_rolling_stats.columns
        ]
        avg = avg.merge(opp_rolling_stats, left_index=True, right_index=True)
    return avg
