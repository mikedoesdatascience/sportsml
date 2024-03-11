from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection

from ..utils.stats import process_averages


def predict_with_uncertainty(rf: sklearn.ensemble.RandomForestRegressor, X: np.array):
    preds = []
    for estimator in rf.estimators_:
        preds.append(estimator.predict(X))
    preds = np.stack(preds)
    return preds.mean(axis=0), preds.std(axis=0)


def train_rf(
    games: pd.DataFrame,
    test_size: float,
    stats_columns: List[str],
    game_id_column: str,
    target_column: str,
    season_column: str,
    date_column: str,
    team_column: str,
    home_column: str,
    rolling_windows: List[int],
    random_state: int = 42,
    rf_kwargs: Dict[str, Any] = {},
):
    avgs = process_averages(
        games,
        stats_columns=stats_columns,
        game_id_column=game_id_column,
        season_column=season_column,
        date_column=date_column,
        team_column=team_column,
        rolling_windows=rolling_windows,
    )
    meta_columns = games.drop(stats_columns, axis=1).columns
    f_columns = avgs.drop(meta_columns, axis=1).columns.tolist() + [home_column]

    if test_size > 0:
        train, test = sklearn.model_selection.train_test_split(
            avgs, test_size=test_size
        )
    else:
        train = avgs
        test = None

    train_X = train[f_columns]
    train_y = train[target_column]

    if test is not None:
        test_X = test[f_columns]
        test_y = test[target_column]

    rf = sklearn.ensemble.RandomForestRegressor(**rf_kwargs)
    rf.fit(train_X, train_y)

    if test is None:
        return {"rf": rf}

    preds, stds = predict_with_uncertainty(rf, test_X)

    return {
        "rf": rf,
        "rmse": sklearn.metrics.root_mean_squared_error(test_y, preds),
        "r2": sklearn.metrics.r2_score(test_y, preds),
        "mae": sklearn.metrics.mean_absolute_error(test_y, preds),
        "accuracy": sklearn.metrics.accuracy_score(test_y > 0, preds > 0),
        "precision": sklearn.metrics.precision_score(test_y > 0, preds > 0),
        "recall": sklearn.metrics.recall_score(test_y > 0, preds > 0),
        "f1": sklearn.metrics.f1_score(test_y > 0, preds > 0),
        "spearmanr": scipy.stats.spearmanr(test_y, preds)[0],
        "pearsonr": scipy.stats.pearsonr(test_y, preds)[0],
    }
