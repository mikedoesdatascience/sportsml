import pandas as pd

from .dataset import CFBGraphDataset
from .features import GRAPH_FEATURES
from .nodes import team_abr_map
from ...mongo import client


def get_team_abr_map():
    return team_abr_map


def get_games(query={}):
    df = pd.DataFrame(client.cfb.games.find(query)).sort_values(["season", "week"])
    df = df[~df[GRAPH_FEATURES].isna().any(axis=1)]
    df["won"] = df["result"] > 0
    df["home"] = df["home"].astype(bool)
    return df


def get_latest_graph():
    games = get_games({"season": max(client.cfb.games.distinct("season"))})
    ds = CFBGraphDataset(games)
    return ds[-1]
