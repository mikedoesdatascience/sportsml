TEAM_STATS_COLUMNS = [
    "Win",
    "Score",
    "PlusMinus",
    "FGM",
    "FGA",
    "FGM3",
    "FGA3",
    "FTM",
    "FTA",
    "OR",
    "DR",
    "Ast",
    "TO",
    "Stl",
    "Blk",
    "PF",
]

OPP_TEAM_STATS_COLUMNS = [f"{col}_OPP" for col in TEAM_STATS_COLUMNS]

STATS_COLUMNS = TEAM_STATS_COLUMNS + OPP_TEAM_STATS_COLUMNS

GRAPH_STATS_COLUMNS = TEAM_STATS_COLUMNS + OPP_TEAM_STATS_COLUMNS

CATEGORICAL_COLUMNS = []

TARGET_COLUMN = "PlusMinus"
SEASON_COLUMN = "Season"
DATE_COLUMN = "DayNum"
TEAM_COLUMN = "TeamID"
TEAM_OPP_COLUMN = "TeamID_OPP"
