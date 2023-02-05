STATS_COLUMNS = [
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
    "sack_yards",
    "sack_fumbles",
    "sack_fumbles_lost",
    "passing_air_yards",
    "passing_yards_after_catch",
    "passing_first_downs",
    "passing_epa",
    "passing_2pt_conversions",
    "dakota",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles",
    "rushing_fumbles_lost",
    "rushing_first_downs",
    "rushing_epa",
    "rushing_2pt_conversions",
    "receptions",
    "special_teams_tds",
    "pacr",
    "racr",
]

OPP_STATS_COLUMNS = [f"opp_{col}" for col in STATS_COLUMNS]

FEATURE_COLUMNS = STATS_COLUMNS + OPP_STATS_COLUMNS + [f'{stat}_opp' for stat in STATS_COLUMNS + OPP_STATS_COLUMNS] + ['rest', 'opp_rest']