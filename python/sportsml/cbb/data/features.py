STATS_COLUMNS = [
    'PTS', 'OPP_PTS', 'FG', 'FGA',
    'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'TRB', 'AST',
    'STL', 'BLK', 'TOV', 'PF', 'OPP_FG', 'OPP_FGA', 'OPP_FG%', 'OPP_3P',
    'OPP_3PA', 'OPP_3P%', 'OPP_FT', 'OPP_FTA', 'OPP_FT%', 'OPP_ORB',
    'OPP_TRB', 'OPP_AST', 'OPP_STL', 'OPP_BLK', 'OPP_TOV', 'OPP_PF',
    'ORtg', 'DRtg', 'Pace', 'FTr', '3PAr', 'TS%', 'TRB%', 'AST%', 'STL%',
    'BLK%', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA', 'DRB%'
]


OPP_STATS_COLUMNS = [f"{col}_OPP" for col in STATS_COLUMNS]

FEATURE_COLUMNS = STATS_COLUMNS + OPP_STATS_COLUMNS + [f'OPP_{stat}' for stat in STATS_COLUMNS + OPP_STATS_COLUMNS]

TEAM = 'TEAM'
OPP_TEAM = 'OPP_TEAM'
DATE = 'DATE'
SEASON = 'SEASON'
POINTS = 'PTS'
OPP_POINTS = 'OPP_PTS'
