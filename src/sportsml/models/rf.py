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
    stats_columns: List[str],
    target_column: str,
    season_column: str,
    date_column: str,
    team_column: str,
    team_opp_column: str,
    test_size: float = 0.2,
    rolling_windows: List[int] = None,
    random_state: int = 42,
    rf_kwargs: Dict[str, Any] = {},
    print_metrics: bool = False,
):
    avgs = process_averages(
        games,
        stats_columns=stats_columns,
        season_column=season_column,
        date_column=date_column,
        team_column=team_column,
        team_opp_column=team_opp_column,
        rolling_windows=rolling_windows,
    ).dropna()

    y = games.loc[avgs.index, target_column].values
    X = avgs.values

    if test_size > 0:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        X_train = X
        y_train = y
        X_test = None
        y_test = None

    rf = sklearn.ensemble.RandomForestRegressor(**rf_kwargs)
    rf.fit(X_train, y_train)

    if X_test is None:
        return {"rf": rf}

    preds, stds = predict_with_uncertainty(rf, X_test)

    metrics = {
        "rmse": sklearn.metrics.root_mean_squared_error(y_test, preds),
        "r2": sklearn.metrics.r2_score(y_test, preds),
        "mae": sklearn.metrics.mean_absolute_error(y_test, preds),
        "accuracy": sklearn.metrics.accuracy_score(y_test > 0, preds > 0),
        "precision": sklearn.metrics.precision_score(y_test > 0, preds > 0),
        "recall": sklearn.metrics.recall_score(y_test > 0, preds > 0),
        "f1": sklearn.metrics.f1_score(y_test > 0, preds > 0),
        "spearmanr": scipy.stats.spearmanr(y_test, preds)[0],
        "pearsonr": scipy.stats.pearsonr(y_test, preds)[0],
    }

    if print_metrics:
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

    return {
        "rf": rf,
        "preds": preds,
        "stds": stds,
        "metrics": metrics,
    }
