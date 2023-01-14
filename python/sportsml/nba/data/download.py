import datetime
import pandas as pd
import time
import tqdm
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from pymongo import ReplaceOne

from .utils import process_games
from ...mongo import client

def download_games():
    games = []
    for team in tqdm.tqdm(teams.get_teams()):
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team["id"])
        games.append(gamefinder.get_data_frames()[0])
        # try not to overload API service
        time.sleep(0.5)
    return pd.concat(games)


def games_from_last_date():
    games = []
    last_date = datetime.date.fromisoformat(
        client.nba.games.find({}).sort('GAME_DATE', -1).limit(1).next()['GAME_DATE']
    ).strftime('%m/%d/%Y')
    for team in tqdm.tqdm(teams.get_teams()):
        gamefinder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team["id"],
            date_from_nullable=last_date
        )
        games.append(gamefinder.get_data_frames()[0])
        time.sleep(0.5)
    return pd.concat(games)


def mongo_upload():
    games = games_from_last_date()
    games = process_games(games)
    games["_id"] = games[["GAME_ID", "TEAM_ID"]].agg(
        lambda x: ".".join(map(str, x)), axis=1
    )
    updates = [
        ReplaceOne({"_id": game["_id"]}, game, upsert=True)
        for game in games.to_dict(orient="records")
    ]
    result = client.nba.games.bulk_write(updates)
