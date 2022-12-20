from typing import List

import pymongo


def group_aggregation(stats: List[str], team_key: str):
    stats_agg = {col: {"$avg": f"${col}"} for col in stats}
    group_agg = {"$group": {"_id": f"${team_key}", "count": {"$count": {}}}}
    group_agg["$group"].update(stats_agg)
    return group_agg


def averages(
    collection: pymongo.collection.Collection,
    stats: List[str],
    team_key: str,
    season: int,
    season_key: str,
    date: str,
    date_key: str,
):
    stats_agg = {col: {"$avg": f"${col}"} for col in stats}
    group_agg = {"$group": {"_id": f"${team_key}", "count": {"$count": {}}}}
    group_agg["$group"].update(stats_agg)
    return list(
        collection.aggregate(
            [
                {"$match": {season_key: season, date_key: {"$lt": date}}},
                group_agg,
            ]
        )
    )