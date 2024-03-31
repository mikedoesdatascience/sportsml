import datetime

import pandas as pd

from .client import OddsAPIClient
from ..mongo import client


def mongo_upload():
    odds_client = OddsAPIClient()
    sports = pd.DataFrame(odds_client.sports())

    if "basketball_nba" in sports["key"].values:
        nba_odds = odds_client.odds("basketball_nba")
        client.nba.odds.insert_many(nba_odds)

    if "americanfootball_nfl" in sports["key"].values:
        nfl_odds = odds_client.odds("americanfootball_nfl")
        client.nfl.odds.insert_many(nfl_odds)

    if "basketball_ncaab" in sports["key"].values:
        cbb_odds = odds_client.odds("basketball_ncaab")
        client.cbb.odds.insert_many(cbb_odds)

    if "americanfootball_ncaaf" in sports["key"].values:
        cfb_odds = odds_client.odds("americanfootball_ncaaf")
        client.cfb.odds.insert_many(cfb_odds)


def get_lines(sport, query={}):
    lines = pd.DataFrame(
        client[sport].odds.aggregate(
            [
                {"$match": query},
                {"$unwind": "$bookmakers"},
                {"$unwind": "$bookmakers.markets"},
            ]
        )
    )
    return lines


def lines_to_dataframe(lines):
    lines = pd.concat(
        [lines.drop("bookmakers", axis=1), pd.DataFrame(lines["bookmakers"].tolist())],
        axis=1,
    )
    lines = lines.rename(columns={"key": "bookmaker"}).drop("last_update", axis=1)

    lines = pd.concat(
        [lines.drop("markets", axis=1), pd.DataFrame(lines["markets"].tolist())], axis=1
    )
    lines = lines.rename(columns={"key": "market"})
    return lines


def get_market_on_date(
    sport: str,
    market: str,
    bookmaker: str = "draftkings",
    game_date_str: str = datetime.date.today().isoformat(),
    market_date_str: str = datetime.date.today().isoformat(),
):
    odds = lines_to_dataframe(
        get_lines(
            sport,
            {
                "commence_time": {"$regex": f"^{game_date_str}"},
                "bookmakers.last_update": {"$regex": f"^{market_date_str}"},
            },
        )
    )
    df = (
        odds[(odds["market"] == market) & (odds["bookmaker"] == bookmaker)]
        .sort_values("commence_time")
        .drop_duplicates(["home_team", "away_team"], keep="last")
    ).reset_index(drop=True)
    if market == "spreads":
        df = pd.concat(
            [
                df.drop("outcomes", axis=1),
                pd.DataFrame(df.apply(format_spreads, axis=1).tolist()),
            ],
            axis=1,
        )
    if market == "h2h":
        df = pd.concat(
            [
                df.drop("outcomes", axis=1),
                pd.DataFrame(df.apply(format_h2h, axis=1).tolist()),
            ],
            axis=1,
        )
    if market == "totals":
        df = pd.concat(
            [
                df.drop("outcomes", axis=1),
                pd.DataFrame(df.apply(format_totals, axis=1).tolist()),
            ],
            axis=1,
        )
    return df


def format_totals(row):
    res = {}
    for val in row["outcomes"]:
        res[f'{val["name"].lower()}_price'] = val["price"]
        res["over_under"] = val["point"]
    return res


def format_spreads(row):
    res = {}
    for outcome in row["outcomes"]:
        if outcome["name"] == row["home_team"]:
            res["home_spread_price"] = outcome["price"]
            res["home_spread"] = outcome["point"]
        if outcome["name"] == row["away_team"]:
            res["away_spread_price"] = outcome["price"]
            res["away_spread"] = outcome["point"]
    return res


def format_h2h(row):
    res = {}
    for outcome in row["outcomes"]:
        if outcome["name"] == row["home_team"]:
            res["home_h2h_price"] = outcome["price"]
        if outcome["name"] == row["away_team"]:
            res["away_h2h_price"] = outcome["price"]
    return res
