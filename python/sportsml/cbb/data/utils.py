import pandas as pd

from .datamodule import CBBGraphDataset
from ...mongo import client


def get_games(query={}):
    df = pd.DataFrame(client.cbb.games.find(query)).sort_values(['Season', 'DayNum'])
    return df


def get_latest_graph():
    games = get_games({'Season': max(client.cbb.games.distinct('Season'))})
    ds = CBBGraphDataset(games)
    return ds[-1]

