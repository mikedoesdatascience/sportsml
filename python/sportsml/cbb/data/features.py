STATS_COLUMNS = [
    "Score",
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

OPP_STATS_COLUMNS = [f"{col}_OPP" for col in STATS_COLUMNS]

GRAPH_FEATURES = STATS_COLUMNS + OPP_STATS_COLUMNS
