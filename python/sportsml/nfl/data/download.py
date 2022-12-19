import pandas as pd
from pymongo import ReplaceOne

from .utils import merge_games_schedule
from ...mongo import client


def get_play_by_play():
    data = pd.read_parquet(
        r"https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.parquet"
    )
    return data


def get_game_totals():
    data = get_play_by_play()
    game_totals = data.groupby(["recent_team", "season", "week"]).sum(numeric_only=True).reset_index()
    return game_totals


def get_schedule():
    schedule = pd.pandas.read_csv(
        "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
    )
    return schedule


def mongo_upload():
    games = get_game_totals()
    schedule = get_schedule()
    games = merge_games_schedule(games, schedule)
    updates = [
        ReplaceOne({"_id": game["_id"]}, game, upsert=True)
        for game in games.to_dict(orient="records")
    ]
    result = client.nfl.games.bulk_write(updates)
