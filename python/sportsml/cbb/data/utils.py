import pandas as pd

from .dataset import CBBHeteroGraphDataset
from .nodes import team_name_map
from ...mongo import client


def get_team_name_map():
    return team_name_map


def get_games(query={}):
    df = pd.DataFrame(client.cbb.games.find(query)).sort_values(["Season", "DayNum"])
    return df


def get_latest_graph():
    games = get_games({"Season": max(client.cbb.games.distinct("Season"))})
    ds = CBBHeteroGraphDataset(games)
    return ds[-1]
