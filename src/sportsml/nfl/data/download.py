from pathlib import Path

import nflreadpy
import pandas as pd

from .features import STATS_COLUMNS
from .names import move_map
from .nodes import team_abr_lookup


def invert_schedule(schedule: pd.DataFrame):
    home = schedule.copy()
    away = schedule.copy()

    away["result"] *= -1
    away["spread_line"] *= -1
    away["location"] = away["location"].replace("Home", "Away")

    home_col_renamer = {
        col: col.removeprefix("home_") for col in home if col.startswith("home_")
    }
    home_col_renamer.update(
        {col: col.replace("away_", "opp_") for col in home if col.startswith("away_")}
    )
    home = home.rename(columns=home_col_renamer)

    away_col_renamer = {
        col: col.removeprefix("away_") for col in away if col.startswith("away_")
    }
    away_col_renamer.update(
        {col: col.replace("home_", "opp_") for col in away if col.startswith("home_")}
    )
    away = away.rename(columns=away_col_renamer)

    return pd.concat([home, away])


def download(output_file: str = None):
    team_stats = (
        nflreadpy.load_team_stats(seasons=True)
        .to_pandas()
        .rename(columns={"opponent_team": "opp_team"})
    )

    team_stats["team"] = team_stats["team"].replace(move_map)
    team_stats["opp_team"] = team_stats["opp_team"].replace(move_map)

    schedule = nflreadpy.load_schedules(True).to_pandas()
    schedule = invert_schedule(schedule=schedule).drop(columns=["game_id"])

    schedule["team"] = schedule["team"].replace(move_map)
    schedule["opp_team"] = schedule["opp_team"].replace(move_map)

    games = team_stats.merge(
        schedule, on=["season", "week", "team", "opp_team"], how="outer"
    )

    games['game_id'] = games[['game_id', 'old_game_id']].bfill(axis=1)["game_id"]

    games = games.dropna(subset=["game_id"])

    first_game = games.drop_duplicates(subset=["game_id"], keep="first")
    last_game = games.drop_duplicates(subset=["game_id"], keep="last")

    games = pd.concat(
        [
            first_game.merge(
                last_game[STATS_COLUMNS + ["game_id"]]
                .add_prefix("opp_")
                .rename(columns={"opp_game_id": "game_id"}),
                on="game_id",
                how="left",
            ).set_index(first_game.index),
            last_game.merge(
                first_game[STATS_COLUMNS + ["game_id"]]
                .add_prefix("opp_")
                .rename(columns={"opp_game_id": "game_id"}),
                on="game_id",
                how="left",
            ).set_index(last_game.index),
        ]
    )

    games["src"] = games["opp_team"].map(team_abr_lookup)
    games["dst"] = games["team"].map(team_abr_lookup)

    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(output_file, index=False)

    return games
