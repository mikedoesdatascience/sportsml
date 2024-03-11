import itertools

import numpy as np
import pandas as pd
import scipy.stats

from sportsml.models.rf import predict_with_uncertainty, train_rf
from sportsml.cbb.data.utils import get_games
from sportsml.cbb.data.features import GRAPH_FEATURES as stats_columns
from sportsml.cbb.data.nodes import team_name_map, team_name_lookup
from sportsml.utils.stats import process_averages

games = get_games()
games = games[games["Season"] <= 2023]
train = games.drop(games[(games["Season"] == 2023) & (games["DayNum"] > 132)].index)

res = train_rf(
    games=train,
    test_size=0.0,
    stats_columns=stats_columns,
    game_id_column="GameID",
    target_column="PlusMinus",
    season_column="Season",
    date_column="DayNum",
    team_column="Team",
    home_column="Loc",
    rolling_windows=[1, 2, 3, 4, 5],
    random_state=42,
    rf_kwargs={"n_jobs": 4, "verbose": 2, "n_estimators": 100},
)

f_columns = res["rf"].feature_names_in_[: len(stats_columns) * 6]

team_avgs = (
    process_averages(
        games=train,
        stats_columns=stats_columns,
        game_id_column="GameID",
        season_column="Season",
        date_column="DayNum",
        team_column="Team",
        rolling_windows=[1, 2, 3, 4, 5],
        use_all_data=True,
    )
    .sort_values(["Season", "DayNum"])
    .drop_duplicates("Team", keep="last")
    .set_index("Team")
)[f_columns]

combinations = np.array(list(itertools.product(team_avgs.index, repeat=2)))

X = np.hstack([team_avgs.loc[combinations[:, 0]], team_avgs.loc[combinations[:, 1]]])

pred_X = pd.DataFrame(X, columns=res["rf"].feature_names_in_[:-1])
pred_X["Loc"] = 0

preds = pd.DataFrame(combinations, columns=["Team", "Team_OPP"])
y, y_std = predict_with_uncertainty(res["rf"], pred_X)

preds["WinProb"] = 1 - scipy.stats.norm(loc=y, scale=y_std).cdf(0)

probs = preds.pivot_table(index="Team", columns="Team_OPP", values="WinProb")

probs = probs.rename(columns=team_name_lookup, index=team_name_lookup)
probs.index = probs.index + 1101
probs.columns = probs.columns + 1101
