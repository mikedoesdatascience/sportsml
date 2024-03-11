import pandas as pd
import pymongo

from .features import STATS_COLUMNS, OPP_STATS_COLUMNS
from .nodes import team_name_map
from ...mongo import client


def format_games(games):
    """Formats kaggle data into mongodb formatted data. Download file with
    `kaggle competitions download -c march-machine-learning-mania-2023`"""
    col_renamer = {col: col.lstrip("W") for col in games.columns if col.startswith("W")}
    col_renamer.update(
        {col: col.lstrip("L") + "_OPP" for col in games.columns if col.startswith("L")}
    )

    games = games.rename(columns=col_renamer)
    games["Loc"] = games["Loc"].map({"H": 1, "A": -1, "N": 0})

    games["TeamID"] = games["TeamID"] - 1101
    games["TeamID_OPP"] = games["TeamID_OPP"] - 1101

    games["GameID"] = (
        games[["Season", "DayNum", "TeamID", "TeamID_OPP"]]
        .astype(str)
        .agg(".".join, axis=1)
    )

    opp_games = games.copy()[["Season", "DayNum", "NumOT", "GameID"]]
    opp_games["Loc"] = -1 * games["Loc"]
    opp_games[["TeamID", "TeamID_OPP"]] = games[["TeamID_OPP", "TeamID"]].values
    opp_games[STATS_COLUMNS] = games[OPP_STATS_COLUMNS].values
    opp_games[OPP_STATS_COLUMNS] = games[STATS_COLUMNS].values

    games = pd.concat([games, opp_games], ignore_index=True)

    games["PlusMinus"] = games["Score"] - games["Score_OPP"]
    games["Won"] = games["PlusMinus"] > 0

    games["target"] = games["TeamID"]
    games["src"] = games["TeamID_OPP"]

    games["Team"] = games["TeamID"].map(team_name_map)
    games["Team_OPP"] = games["TeamID_OPP"].map(team_name_map)

    games["_id"] = games[["Season", "DayNum", "TeamID", "TeamID_OPP"]].agg(
        lambda x: ".".join(map(str, x)), axis=1
    )

    return games


def mongo_upload(games):
    updates = [
        pymongo.ReplaceOne({"_id": game["_id"]}, game, upsert=True)
        for game in games.to_dict(orient="records")
    ]
    _ = client.cbb.games.bulk_write(updates)
    return


if __name__ == "__main__":
    mongo_upload(format_games(pd.read_csv("data/MNCAATourneyDetailedResults.csv")))
    mongo_upload(format_games(pd.read_csv("data/MRegularSeasonDetailedResults.csv")))
