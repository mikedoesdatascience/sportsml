import datetime
import time
from pathlib import Path

import pandas as pd
import tqdm
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

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
