import pandas as pd

from ...mongo import client


def get_games(query={}):
    df = pd.DataFrame(client.cbb.games.find(query)).sort_values(['Season', 'DayNum'])
    return df