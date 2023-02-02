import pandas as pd

from .features import STATS_COLUMNS, OPP_STATS_COLUMNS
from .nodes import team_idx_map
from ...mongo import client, group_aggregation


def process_games(games: pd.DataFrame):
    games = games.dropna()
    games = games[games.groupby("GAME_ID")["GAME_ID"].transform("count") == 2]

    first_game = games.drop_duplicates(subset=["GAME_ID"], keep="first")
    last_game = games.drop_duplicates(subset=["GAME_ID"], keep="last")

    games = pd.concat(
        [
            first_game.merge(
                last_game[STATS_COLUMNS + ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID"]],
                on="GAME_ID",
                how="left",
                suffixes=("", "_OPP"),
            ).set_index(first_game.index),
            last_game.merge(
                first_game[STATS_COLUMNS + ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID"]],
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
    games["target"] = games["TEAM_ID"].map(team_idx_map)

    games["SEASON"] = games["SEASON_ID"].astype(str).str.slice(start=1).astype(int)

    return games


def process_averages(games):
    games = games.sort_values('GAME_DATE')
    avg = games.copy().drop(STATS_COLUMNS+OPP_STATS_COLUMNS, axis=1)
    avg_stats = games.groupby(['SEASON_ID', 'TEAM_ABBREVIATION'])[STATS_COLUMNS + OPP_STATS_COLUMNS].expanding().mean().groupby(['SEASON_ID', 'TEAM_ABBREVIATION']).shift(1).droplevel([0, 1])
    return avg.merge(avg_stats, left_index=True, right_index=True)


def get_regular_season_games():
    return pd.DataFrame(
        client.nba.games.find({
            'SEASON_ID': {'$regex': '^2'},
            'GAME_ID': {'$regex': '^0'},
        })
    )


def get_regular_season_averages(date):
    season_id = client.nba.games.find_one({'GAME_DATE': date}).get('SEASON_ID')
    result = list(
        client.nba.games.aggregate(
            [
                {
                    "$match": {
                        'SEASON_ID': season_id,
                        'GAME_DATE': {"$lt": date},
                        'GAME_ID': {'$regex': '^0'}
                    }
                },
                group_aggregation(STATS_COLUMNS + OPP_STATS_COLUMNS, 'TEAM_ABBREVIATION')
            ]
        )
    )
    return {
        res.pop('_id'): {
            'stats': {
                k: v
                for k,v in res.items()
                if '_OPP' not in k
            },
            'opp_stats': {
                k: v
                for k,v in res.items()
                if '_OPP' in k
            }
        }
        for res in result
    }


def featurize_games(avgs):
    first_avgs = avgs.drop_duplicates('GAME_ID', keep='first').set_index('GAME_ID', drop=True)
    last_avgs = avgs.drop_duplicates('GAME_ID', keep='last').set_index('GAME_ID', drop=True)

    stats = STATS_COLUMNS + OPP_STATS_COLUMNS
    opp_stats = [f'OPP_{stat}' for stat in stats]

    first_avgs[opp_stats] = last_avgs[stats]

    return first_avgs