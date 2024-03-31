import pandas as pd
from sklearn.model_selection import train_test_split

from .datamodule import NBAGraphDataset
from .features import STATS_COLUMNS, OPP_STATS_COLUMNS, FEATURE_COLUMNS
from .nodes import team_idx_map, team_abr_map
from ...mongo import client, group_aggregation


def get_team_abr_map():
    return team_abr_map


def process_games(games: pd.DataFrame):
    games = games.dropna()
    games = games[games.groupby("GAME_ID")["GAME_ID"].transform("count") == 2]

    first_game = games.drop_duplicates(subset=["GAME_ID"], keep="first")
    last_game = games.drop_duplicates(subset=["GAME_ID"], keep="last")

    games = pd.concat(
        [
            first_game.merge(
                last_game[
                    STATS_COLUMNS
                    + ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID"]
                ],
                on="GAME_ID",
                how="left",
                suffixes=("", "_OPP"),
            ).set_index(first_game.index),
            last_game.merge(
                first_game[
                    STATS_COLUMNS
                    + ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID"]
                ],
                on="GAME_ID",
                how="left",
                suffixes=("", "_OPP"),
            ).set_index(last_game.index),
        ]
    ).reset_index(drop=True)

    games["HOME"] = (games["MATCHUP"].str[4] != "@").astype(float)

    games["GAME_DATE_dt"] = pd.to_datetime(games["GAME_DATE"])
    games["REST"] = (
        games.sort_values("GAME_DATE")
        .groupby(["SEASON_ID", "TEAM_ID"])["GAME_DATE_dt"]
        .transform("diff")
        .dt.days
    )

    games["src"] = games["TEAM_ID_OPP"].map(team_idx_map)
    games["dst"] = games["TEAM_ID"].map(team_idx_map)

    games["SEASON"] = games["SEASON_ID"].astype(str).str.slice(start=1).astype(int)

    return games


def process_averages(games):
    games = games.sort_values("GAME_DATE")
    avg = games.copy().drop(STATS_COLUMNS + OPP_STATS_COLUMNS, axis=1)
    avg_stats = (
        games.groupby(["SEASON", "TEAM_ABBREVIATION"])[
            STATS_COLUMNS + OPP_STATS_COLUMNS
        ]
        .expanding()
        .mean()
        .groupby(["SEASON", "TEAM_ABBREVIATION"])
        .shift(1)
        .droplevel([0, 1])
    )
    rolling_stats = (
        games.groupby(["SEASON", "TEAM_ABBREVIATION"])[
            STATS_COLUMNS + OPP_STATS_COLUMNS
        ]
        .rolling(5, 1, closed="left")
        .mean()
        .droplevel([0, 1])
    )
    rolling_stats.columns = [f"{col}_rolling" for col in rolling_stats.columns]
    return avg.merge(avg_stats, left_index=True, right_index=True).merge(
        rolling_stats, left_index=True, right_index=True
    )


def get_games(query={}):
    query.update(
        {
            "SEASON_ID": {"$regex": "^2|^4"},
            "GAME_ID": {"$regex": "^0"},
        }
    )
    df = pd.DataFrame(client.nba.games.find(query)).sort_values("GAME_DATE")
    df["won"] = df["PTS"] > df["PTS_OPP"]
    df["home"] = df["HOME"].astype(bool)
    df["y"] = df["PTS"] - df["PTS_OPP"]
    return df


def get_regular_season_games(query={}):
    query.update(
        {
            "SEASON_ID": {"$regex": "^2"},
            "GAME_ID": {"$regex": "^0"},
        }
    )
    df = pd.DataFrame(client.nba.games.find(query)).sort_values("GAME_DATE")
    return df


def get_latest_graph():
    query = {
        "SEASON_ID": {"$regex": "^2|^4"},
        "GAME_ID": {"$regex": "^0"},
    }
    games = get_games({"SEASON": max(client.nba.games.find(query).distinct("SEASON"))})
    ds = NBAGraphDataset(games)
    return ds[-1]


def get_season_averages(date, season=None):
    if season is None:
        season = client.nba.games.find_one({"GAME_DATE": date}).get("SEASON")
    result = list(
        client.nba.games.aggregate(
            [
                {
                    "$match": {
                        "SEASON": season,
                        "SEASON_ID": {"$regex": "^2|^4"},
                        "GAME_DATE": {"$lt": date},
                        "GAME_ID": {"$regex": "^0"},
                    }
                },
                group_aggregation(
                    STATS_COLUMNS + OPP_STATS_COLUMNS, "TEAM_ABBREVIATION"
                ),
            ]
        )
    )
    return {
        res.pop("_id"): {
            "stats": {k: v for k, v in res.items() if "_OPP" not in k},
            "opp_stats": {k: v for k, v in res.items() if "_OPP" in k},
        }
        for res in result
    }


def featurize_games(avgs):
    first_avgs = avgs.drop_duplicates("GAME_ID", keep="first").set_index(
        "GAME_ID", drop=True
    )
    last_avgs = avgs.drop_duplicates("GAME_ID", keep="last").set_index(
        "GAME_ID", drop=True
    )

    stats = STATS_COLUMNS + OPP_STATS_COLUMNS
    stats = stats + [f"{col}_rolling" for col in stats]
    opp_stats = [f"OPP_{stat}" for stat in stats]

    first_avgs[opp_stats] = last_avgs[stats]

    return first_avgs


def get_training_data(feature_columns=FEATURE_COLUMNS):
    games = get_regular_season_games()
    avgs = process_averages(games)
    f = featurize_games(avgs)
    f = f.dropna(subset=feature_columns)
    return f


def get_training_data_split(
    feature_columns=FEATURE_COLUMNS,
    target_column="PLUS_MINUS",
    test_size=0.2,
    random_state=None,
):
    f = get_training_data()
    f = f.dropna(subset=FEATURE_COLUMNS)
    X = f[feature_columns].values
    y = f[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
