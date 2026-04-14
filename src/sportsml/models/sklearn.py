from typing import Any, Dict, List

import mlflow.pyfunc
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

from ..utils.stats import process_averages


def predict_with_uncertainty(rf: sklearn.ensemble.RandomForestRegressor, X: np.array):
    preds = []
    for estimator in rf.estimators_:
        preds.append(estimator.predict(X))
    preds = np.stack(preds)
    return preds.mean(axis=0), preds.std(axis=0)


def train_sklearn(
    games: pd.DataFrame,
    model: sklearn.base.BaseEstimator,
    stats_columns: List[str],
    target_column: str,
    season_column: str,
    date_column: str,
    team_column: str,
    team_opp_column: str,
    save_dir: str,
    meta_columns: List[str] = None,
    categorical_columns: List[str] = None,
    train_seasons: list[int] = None,
    test_seasons: list[int] = None,
    rolling_windows: List[int] = None,
    random_state: int = 42,
    print_metrics: bool = False,
):
    unique_seasons = sorted(games[season_column].unique(), reverse=True)

    if test_seasons is None:
        test_seasons = unique_seasons[:1]

    if train_seasons is None:
        train_seasons = [s for s in unique_seasons if s not in test_seasons]

    avgs = process_averages(
        games,
        stats_columns=stats_columns,
        season_column=season_column,
        date_column=date_column,
        team_column=team_column,
        team_opp_column=team_opp_column,
        rolling_windows=rolling_windows,
    ).dropna()

    X = avgs.values
    y = games.loc[avgs.index, target_column].values

    if meta_columns:
        meta_data = games.loc[avgs.index, meta_columns]
        X_meta = KNNImputer(n_neighbors=10).fit_transform(meta_data.values)
        X = np.hstack([X, X_meta])

    if categorical_columns:
        cat_data = games.loc[avgs.index, categorical_columns]
        X_cat = OneHotEncoder(
            sparse_output=False, min_frequency=10, handle_unknown="ignore"
        ).fit_transform(cat_data.values)
        X = np.hstack([X, X_cat])

    if test_seasons:
        X_train = X[
            games.loc[avgs.index, season_column].isin(
                [s for s in unique_seasons if s not in test_seasons]
            )
        ]
        y_train = y[
            games.loc[avgs.index, season_column].isin(
                [s for s in unique_seasons if s not in test_seasons]
            )
        ]
        X_test = X[games.loc[avgs.index, season_column].isin(test_seasons)]
        y_test = y[games.loc[avgs.index, season_column].isin(test_seasons)]
    else:
        X_train = X
        y_train = y
        X_test = None
        y_test = None

    model.fit(X_train, y_train)

    if X_test is None:
        return {"model": model}

    if isinstance(model, sklearn.ensemble.RandomForestRegressor):
        preds, stds = predict_with_uncertainty(model, X_test)
    else:
        preds = model.predict(X_test)
        stds = None

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

    season = games[games[season_column] == games[season_column].max()]
    avgs = process_averages(
        season,
        stats_columns=stats_columns,
        season_column=season_column,
        date_column=date_column,
        team_column=team_column,
        team_opp_column=team_opp_column,
        rolling_windows=rolling_windows,
    )
    team_stats = season.sort_values(
        [season_column, date_column]
    ).drop_duplicates(team_column, keep="last")

    team_features = avgs.loc[team_stats.index]
    team_features = team_features[[col for col in team_features.columns if not col.startswith('opp_')]]
    team_features.index = team_features.index.map(team_stats[team_column])

    predictor = SportsMLPredictor(model=model, team_features=team_features)

    mlflow.pyfunc.save_model(save_dir, python_model=predictor)

    return {
        "model": model,
        "preds": preds,
        "stds": stds,
        "metrics": metrics,
    }


class SportsMLPredictor(mlflow.pyfunc.PythonModel):
    """
    MLFlow Pyfunc model wrapper for sports prediction.

    This model stores processed statistics per team and generates predictions
    for pairs of teams using a fitted sklearn model.

    Attributes:
        model: The fitted sklearn model for predictions
        team_stats: DataFrame containing processed statistics per team
        meta_columns: List of metadata column names used in training
        categorical_columns: List of categorical column names used in training
        rolling_windows: List of rolling window sizes used in training
        stats_columns: List of statistics column names used in training
    """

    def __init__(
        self,
        model: Any,
        team_features: pd.DataFrame,
    ):
        """
        Initialize the SportsML predictor.

        Args:
            model: Fitted sklearn model
            team_features: DataFrame with team statistics (indexed by team ID)
        """
        self.model = model
        self.team_features = team_features

    def long_to_wide(
        self,
        preds: pd.DataFrame,
        prob: bool = False,
        team_name_map: dict[int, str] = None,
        sorted: bool = False,
    ):
        preds = preds.pivot_table(
            values="prob" if prob else "preds", index="team", columns="opp"
        )
        if team_name_map:
            preds = preds.rename(index=team_name_map, columns=team_name_map)
        if sorted:
            preds = preds.loc[
                preds.mean(axis=0).sort_values().index,
                preds.mean(axis=0).sort_values().index,
            ]
        return preds

    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for team pairs.

        Args:
            context: MLFlow context (unused but required by PythonModel interface)
            model_input: DataFrame with columns [team_id, team_opp_id] or similar.
                        Column names should match what was used during training.
                        Can also include: season, date for filtering team stats.

        Returns:
            Array of predictions (one per row in model_input)
        """
        X = np.hstack(
            [
                self.team_features.loc[model_input.team],
                self.team_features.loc[model_input.opp],
            ]
        )

        if isinstance(self.model, sklearn.ensemble.RandomForestRegressor):
            preds, stds = predict_with_uncertainty(self.model, X)
            result = model_input.assign(preds=preds, std=stds)
        else:
            preds = self.model.predict(X)
            result = model_input.assign(preds=preds)
        result['prob'] = scipy.stats.norm.cdf(result['preds'] / result['preds'].std())
        return result
