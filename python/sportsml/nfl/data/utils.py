import pandas as pd

from ...mongo import client, group_aggregation
from .features import OPP_STATS_COLUMNS, STATS_COLUMNS
from .names import move_map


def merge_games_schedule(games, schedule):
    games = games.rename(columns={"recent_team": "team"})
    games = games.set_index(["season", "week", "team"])

    schedule.home_team = schedule.home_team.replace(move_map)
    schedule.away_team = schedule.away_team.replace(move_map)

    schedule.game_id = (
        schedule[["season", "week", "away_team", "home_team"]]
        .astype(str)
        .agg("-".join, axis=1)
    )

    schedule_home = schedule.copy()
    schedule_home.columns = schedule_home.columns.map(lambda x: x.replace("home_", ""))
    schedule_home.columns = schedule_home.columns.map(
        lambda x: x.replace("away_", "opp_")
    )
    schedule_home.home = 1
    schedule_home = schedule_home.set_index(["season", "week", "team"])

    schedule_away = schedule.copy()
    schedule_away.columns = schedule_away.columns.map(lambda x: x.replace("away_", ""))
    schedule_away.columns = schedule_away.columns.map(
        lambda x: x.replace("home_", "opp_")
    )
    schedule_away.home = 0
    schedule_away = schedule_away.set_index(["season", "week", "team"])

    games_home = games.merge(
        schedule_home, how="right", left_index=True, right_index=True
    )

    games_away = games.merge(
        schedule_away, how="right", left_index=True, right_index=True
    )

    games_home = games_home.dropna(subset=["result"]).reset_index()
    games_away = games_away.dropna(subset=["result"]).reset_index()

    games = pd.concat([games_home, games_away], ignore_index=True).reset_index(
        drop=True
    )

    games.result = games.score - games.opp_score

    games["_id"] = (
        games[["season", "week", "team", "opp_team"]].astype(str).agg("-".join, axis=1)
    )

    first_game = games.drop_duplicates(subset=["game_id"], keep="first")
    last_game = games.drop_duplicates(subset=["game_id"], keep="last")

    games = pd.concat(
        [
            first_game.merge(
                last_game[STATS_COLUMNS + ["game_id"]]
                .add_prefix("opp_")
                .rename(columns={"opp_game_id": "game_id"}),
                on="game_id",
                how="left",
            ).set_index(first_game.index),
            last_game.merge(
                first_game[STATS_COLUMNS + ["game_id"]]
                .add_prefix("opp_")
                .rename(columns={"opp_game_id": "game_id"}),
                on="game_id",
                how="left",
            ).set_index(last_game.index),
        ]
    )

    games['home'] = games.apply(lambda x: int(x['game_id'].endswith(x['team'])), axis=1)

    return games


def get_season_averages(season, week):
    result = list(
        client.nfl.games.aggregate(
            [
                {
                    "$match": {
                        'season': season,
                        'week': {"$lt": week}
                    }
                },
                group_aggregation(STATS_COLUMNS + OPP_STATS_COLUMNS, 'team')
            ]
        )
    )
    return {
        res.pop('_id'): {
            'stats': {
                k: v
                for k,v in res.items()
                if 'opp_' not in k
            },
            'opp_stats': {
                k: v
                for k,v in res.items()
                if 'opp_' in k
            }
        }
        for res in result
    }


def process_averages(games):
    games = games.sort_values(['season', 'week'])
    avg = games.copy().drop(STATS_COLUMNS+OPP_STATS_COLUMNS, axis=1)
    avg_stats = games.groupby(['season', 'team'])[STATS_COLUMNS + OPP_STATS_COLUMNS].expanding().mean().groupby(['season', 'team']).shift(1).droplevel([0, 1])
    return avg.merge(avg_stats, left_index=True, right_index=True)

def featurize_games(avgs):
    avgs = avgs.copy()
    opp_avgs = avgs.copy()
    opp_avgs['_id'] = opp_avgs.apply(
        lambda row: '-'.join([
            str(row['season']), 
            str(row['week']), 
            row['opp_team'], 
            row['team']
        ]), 
        axis=1
    )

    stats = STATS_COLUMNS + OPP_STATS_COLUMNS

    opp_stats = [f'{stat}_opp' for stat in stats]

    avgs = avgs.set_index('_id')
    opp_avgs = opp_avgs.set_index('_id')

    avgs[opp_stats] = opp_avgs[stats]

    return avgs