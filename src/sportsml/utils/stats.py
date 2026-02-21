from typing import List

import pandas as pd


def process_averages(
    games: pd.DataFrame,
    stats_columns: List[str],
    game_meta_columns: List[str] = None,
    season_column: str = "season",
    date_column: str = "date",
    team_column: str = "team",
    team_opp_column: str = "team_opp",
    rolling_windows: List[int] = None,
    use_all_data: bool = False,
    avg_suffix: str = "_avg",
    rolling_suffix: str = "_rolling",
    opp_prefix: str = "opp_",
) -> pd.DataFrame:
    game_meta_columns = game_meta_columns or []
    
    shift = 0 if use_all_data else 1
    closed = "right" if use_all_data else "left"

    rolling_windows = rolling_windows or []

    games = games.sort_values([season_column, date_column])
    avg = games[game_meta_columns].copy()

    expanding_stats = (
        games.groupby([season_column, team_column])[stats_columns]
        .expanding()
        .mean()
        .groupby([season_column, team_column])
        .shift(shift)
        .droplevel([0, 1])
    )
    avg = avg.join(expanding_stats.add_suffix(avg_suffix), how="left")

    expanding_stats_opp = (
        games.groupby([season_column, team_opp_column])[stats_columns]
        .expanding()
        .mean()
        .groupby([season_column, team_opp_column])
        .shift(shift)
        .droplevel([0, 1])
    )
    avg = avg.join(
        expanding_stats_opp.add_prefix(opp_prefix).add_suffix(avg_suffix), how="left"
    )

    for rolling_window in rolling_windows:
        rolling_stats = (
            games.groupby([season_column, team_column])[stats_columns]
            .rolling(rolling_window, 1, closed=closed)
            .mean()
            .droplevel([0, 1])
        )
        avg = avg.join(
            rolling_stats.add_suffix(f"{rolling_suffix}_{rolling_window}"), how="left"
        )

        rolling_stats_opp = (
            games.groupby([season_column, team_opp_column])[stats_columns]
            .rolling(rolling_window, 1, closed=closed)
            .mean()
            .droplevel([0, 1])
        )
        avg = avg.join(
            rolling_stats_opp.add_prefix(opp_prefix).add_suffix(
                f"{rolling_suffix}_{rolling_window}"
            ),
            how="left",
        )

    return avg
