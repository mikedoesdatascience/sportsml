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

CATEGORICAL_COLUMNS = [
    "team",
    "opp_team",
    "game_type",
    "weekday",
    "gametime",
    "location",
    "div_game",
    "roof",
    "surface",
    "qb_name",
    "opp_qb_name",
    "coach",
    "opp_coach",
    "referee",
    "stadium",
]

META_COLUMNS = [
    "rest",
    "opp_rest",
    "moneyline",
    "opp_moneyline",
    "spread_line",
    "spread_odds",
    "opp_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
    "temp",
    "wind",
]

FEATURE_COLUMNS = (
    STATS_COLUMNS
    + OPP_STATS_COLUMNS
    + [f"{stat}_opp" for stat in STATS_COLUMNS + OPP_STATS_COLUMNS]
    + ["home", "rest", "opp_rest"]
)

GRAPH_FEATURES = STATS_COLUMNS + OPP_STATS_COLUMNS

TARGET_COLUMN = "result"
SEASON_COLUMN = "season"
DATE_COLUMN = "week"
TEAM_COLUMN = "team"
TEAM_OPP_COLUMN = "opp_team"