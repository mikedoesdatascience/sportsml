import datetime
from pathlib import Path

import pandas as pd
import time
import tqdm
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder

from .utils import process_games


def download(output_file: str = None):
    games = []
    for team in tqdm.tqdm(teams.get_teams()):
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team["id"])
        games.append(gamefinder.get_data_frames()[0])
        # try not to overload API service
        time.sleep(0.5)
    games = pd.concat(games)
    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(output_file, index=False)
    return games


def games_from_last_date():
    games = []
    last_date = datetime.date.fromisoformat(
        client.nba.games.find({}).sort("GAME_DATE", -1).limit(1).next()["GAME_DATE"]
    ).strftime("%m/%d/%Y")
    for team in tqdm.tqdm(teams.get_teams()):
        gamefinder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team["id"], date_from_nullable=last_date
        )
        games.append(gamefinder.get_data_frames()[0])
        time.sleep(0.5)
    return pd.concat(games)
