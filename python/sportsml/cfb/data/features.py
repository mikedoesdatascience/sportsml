STATS_COLUMNS = [
    "rushingTDs",
    "passingTDs",
    "kickReturnYards",
    "kickReturnTDs",
    "kickReturns",
    "kickingPoints",
    "interceptionYards",
    "interceptionTDs",
    "passesIntercepted",
    "fumblesRecovered",
    "totalFumbles",
    "possessionTime",
    "interceptions",
    "fumblesLost",
    "turnovers",
    "yardsPerRushAttempt",
    "rushingAttempts",
    "rushingYards",
    "yardsPerPass",
    "netPassingYards",
    "totalYards",
    "firstDowns",
    "totalPenalties",
    "totalPenaltiesYards",
    "passingCompletions",
    "passingAttempts",
    "fourthDownAtt",
    "fourthDownConv",
    "thirdDownAtt",
    "thirdDownConv",
]

OPP_STATS_COLUMNS = [f"opp_{col}" for col in STATS_COLUMNS]

FEATURE_COLUMNS = (
    STATS_COLUMNS
    + OPP_STATS_COLUMNS
    + [f"{stat}_opp" for stat in STATS_COLUMNS + OPP_STATS_COLUMNS]
    + ["home"]
)

GRAPH_FEATURES = STATS_COLUMNS + OPP_STATS_COLUMNS
