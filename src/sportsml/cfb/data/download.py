import datetime
import os

import httpx
import pandas as pd
import tqdm
from pymongo import ReplaceOne

from .nodes import team_abr_lookup
from ...mongo import client


GAME_URL = "https://api.collegefootballdata.com/games/teams"
CALENDAR_URL = "https://api.collegefootballdata.com/calendar"
CFBD_API_KEY = os.getenv("CFBD_API_KEY")


def calendar(year: int):
    resp = httpx.get(
        CALENDAR_URL,
        params={"year": year},
        headers={"Authorization": f"Bearer {CFBD_API_KEY}"},
    )
    resp.raise_for_status()
    return resp.json()


def download_games(year: int, week: int, season_type="regular"):
    resp = httpx.get(
        GAME_URL,
        params={"year": year, "week": week, "seasonType": season_type},
        headers={"Authorization": f"Bearer {CFBD_API_KEY}"},
    )
    resp.raise_for_status()
    games = pd.concat([game_to_dataframe(game) for game in resp.json()])
    games["season"] = year
    games["week"] = week
    games["_id"] = (
        games[["season", "week", "team", "opp_team"]].astype(str).agg("-".join, axis=1)
    )
    return games.reset_index(drop=True).fillna(0.0)


def possession_time(time: str):
    minutes, seconds = time.split(":")
    return int(minutes) + int(seconds) / 60


def game_to_dataframe(game):
    stats = []
    for team in game["teams"]:
        df = pd.DataFrame(team["stats"]).set_index("category").T

        if "possessionTime" in df:
            df["possessionTime"] = df["possessionTime"].apply(possession_time)

        if "totalPenaltiesYards" in df:
            df["totalPenaltiesYards"] = df["totalPenaltiesYards"].str.replace("--", "-")
            df[["totalPenalties", "totalPenaltiesYards"]] = df.pop(
                "totalPenaltiesYards"
            ).str.split("-", expand=True)

        if "completionAttempts" in df:
            df["completionAttempts"] = df["completionAttempts"].str.replace("--", "-")
            df[["passingCompletions", "passingAttempts"]] = df.pop(
                "completionAttempts"
            ).str.split("-", expand=True)

        if "fourthDownEff" in df:
            df["fourthDownEff"] = df["fourthDownEff"].str.replace("--", "-")
            df[["fourthDownAtt", "fourthDownConv"]] = df.pop("fourthDownEff").str.split(
                "-", expand=True
            )

        if "thirdDownEff" in df:
            df["thirdDownEff"] = df["thirdDownEff"].str.replace("--", "-")
            df[["thirdDownAtt", "thirdDownConv"]] = df.pop("thirdDownEff").str.split(
                "-", expand=True
            )

        df = df.astype(float)
        df["team"] = team["school"]
        df["conference"] = team["conference"]
        df["home"] = int(team["homeAway"] == "home")
        df["points"] = team["points"]
        stats.append(df)
    df = pd.concat(
        [
            pd.concat([stats[0], stats[1].add_prefix("opp_")], axis=1),
            pd.concat([stats[1], stats[0].add_prefix("opp_")], axis=1),
        ],
        axis=0,
    )
    df["result"] = df["points"] - df["opp_points"]
    return df.reset_index(drop=True)


def mongo_upload():
    years = range(2004, datetime.datetime.today().year + 1)
    games = []
    for year in tqdm.tqdm(years):
        for week in calendar(year):
            if "spring" in week["seasonType"]:
                continue
            if datetime.datetime.today().isoformat() < week["firstGameStart"]:
                continue
            try:
                games.append(download_games(year, week["week"], week["seasonType"]))
            except Exception:
                print(week)
    games = pd.concat(games).reset_index(drop=True)
    games["src"] = games["opp_team"].map(team_abr_lookup)
    games["dst"] = games["team"].map(team_abr_lookup)
    updates = [
        ReplaceOne({"_id": game["_id"]}, game, upsert=True)
        for game in games.to_dict(orient="records")
    ]
    _ = client.cfb.games.bulk_write(updates)
    return
