import pandas as pd

from .features import STATS_COLUMNS
from .nodes import team_idx_map


def process_games(games: pd.DataFrame):
    games = games.dropna()
    games = games[games.groupby("GAME_ID")["GAME_ID"].transform("count") == 2]

    # summer league games have 10 digits in GAME_ID
    games = games[games["GAME_ID"].astype(str).apply(len) == 8]

    first_game = games.drop_duplicates(subset=["GAME_ID"], keep="first")
    last_game = games.drop_duplicates(subset=["GAME_ID"], keep="last")

    games = pd.concat(
        [
            first_game.merge(
                last_game[STATS_COLUMNS + ["GAME_ID"]],
                on="GAME_ID",
                how="left",
                suffixes=("", "_OPP"),
            ).set_index(first_game.index),
            last_game.merge(
                first_game[STATS_COLUMNS + ["GAME_ID"]],
                on="GAME_ID",
                how="left",
                suffixes=("", "_OPP"),
            ).set_index(last_game.index),
        ]
    )

    games["HOME"] = (games["MATCHUP"].str[4] != "@").astype(float)

    games["GAME_DATE_dt"] = pd.to_datetime(games["GAME_DATE"])
    games["REST"] = (
        games.sort_values("GAME_DATE")
        .groupby(["SEASON_ID", "TEAM_ID"])["GAME_DATE_dt"]
        .transform("diff")
        .dt.days
    )

    games["src"] = games["TEAM_ID_OPP"].map(team_idx_map)
    games["target"] = games["TEAM_ID"].map(team_idx_map)

    games["SEASON"] = games["SEASON_ID"].astype(str).str.slice(start=1).astype(int)

    return games
