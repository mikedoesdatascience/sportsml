from typing import List

import pandas as pd


def process_averages(
    games: pd.DataFrame,
    stats_columns: List[str],
    game_id_column: str,
    season_column: str,
    date_column: str,
    team_column: str,
    rolling_windows: List[int],
    use_all_data: bool = False,
) -> pd.DataFrame:
    # if we use_all_data then expanding means should not be shifted
    # and rolling means should not be closed
    # use_all_data should be True when generating data for training
    shift = 0 if use_all_data else 1
    closed = None if use_all_data else "left"

    f_columns = []

    games = games.sort_values([season_column, date_column])
    avg = games.copy().drop(stats_columns, axis=1)
    avg_stats = (
        games.groupby([season_column, team_column])[stats_columns]
        .expanding()
        .mean()
        .groupby([season_column, team_column])
        .shift(shift)
        .droplevel([0, 1])
    )
    avg_stats.columns = [f"{col}_avg" for col in avg_stats.columns]
    f_columns += avg_stats.columns.tolist()
    avg = avg.merge(avg_stats, left_index=True, right_index=True)
    for rolling_window in rolling_windows:
        rolling_stats = (
            games.groupby([season_column, team_column])[stats_columns]
            .rolling(rolling_window, 1, closed=closed)
            .mean()
            .droplevel([0, 1])
        )
        rolling_stats.columns = [
            f"{col}_rolling_{rolling_window}" for col in rolling_stats.columns
        ]
        f_columns += rolling_stats.columns.tolist()
        avg = avg.merge(rolling_stats, left_index=True, right_index=True)
    opp_f_columns = [f"OPP_{stat}" for stat in f_columns]

    first = avg.drop_duplicates("GameID", keep="first").set_index("GameID", drop=True)
    last = avg.drop_duplicates("GameID", keep="last").set_index("GameID", drop=True)

    first[opp_f_columns] = last[f_columns]
    last[opp_f_columns] = first[f_columns]
    return pd.concat([first, last]).reset_index()
