import pandas as pd

from .client import OddsAPIClient
from ..mongo import client


def mongo_upload():
    odds_client = OddsAPIClient()
    sports = pd.DataFrame(odds_client.sports())

    if "basketball_nba" in sports["key"].values:
        nba_odds = odds_client.odds("basketball_nba")
        client.nba.odds.insert_many(nba_odds)

    if "americanfootbal_nfl" in sports["key"].values:
        nfl_odds = odds_client.odds("americanfootbal_nfl")
        client.nfl.odds.insert_many(nfl_odds)

    if "basketball_ncaab" in sports["key"].values:
        cbb_odds = odds_client.odds("basketball_ncaab")
        client.cbb.odds.insert_many(cbb_odds)

    if "americanfootball_ncaaf" in sports["key"].values:
        cfb_odds = odds_client.odds("americanfootball_ncaaf")
        client.cfb.odds.insert_many(cfb_odds)
