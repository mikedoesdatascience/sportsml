import jsonargparse
import jsonargparse.typing
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from . import __version__
from .cbb.data import features as cbb_features
from .cbb.data.download import download as cbb_download
from .cfb.data import features as cfb_features
from .cfb.data.download import download as cfb_download
from .graph.fit import fit as pyg_fit
from .models.sklearn import train_sklearn
from .nba.data import features as nba_features
from .nba.data.download import download as nba_download
from .nfl.data import features as nfl_features
from .nfl.data.download import download as nfl_download
from .utils.process import process

jsonargparse.typing.register_type(
    pd.DataFrame, lambda x: getattr(x, "filename"), lambda x: pd.read_csv(x)
)


def version():
    """Print the version of the sportsml package."""
    print(__version__)


def cli():
    jsonargparse.auto_cli(
        {
            "cbb": {
                "download": cbb_download,
                "process": process,
                "pyg": {"fit": pyg_fit},
                "sklearn": {"fit": train_sklearn},
            },
            "cfb": {
                "download": cfb_download,
                "process": process,
                "sklearn": {"fit": train_sklearn},
            },
            "nba": {
                "download": nba_download,
                "process": process,
                "sklearn": {"fit": train_sklearn},
            },
            "nfl": {
                "download": nfl_download,
                "process": process,
                "sklearn": {"fit": train_sklearn},
            },
            "version": version,
        },
        as_positional=False,
        set_defaults={
            "cbb.download.output_file": "data/cbb/raw.csv",
            "cbb.process.games": "data/cbb/raw.csv",
            "cbb.process.stats_columns": cbb_features.GRAPH_STATS_COLUMNS,
            "cbb.process.target_column": cbb_features.TARGET_COLUMN,
            "cbb.process.output_file": "data/cbb/data.csv",
            "cbb.pyg.fit.datamodule": {
                "games": "data/cbb/data.csv",
                "stats_columns": cbb_features.GRAPH_STATS_COLUMNS,
                "target_column": cbb_features.TARGET_COLUMN,
                "season_column": cbb_features.SEASON_COLUMN,
                "date_column": cbb_features.DATE_COLUMN,
            },
            "cbb.pyg.fit.model": {
                "encoder": {
                    "class_path": "sportsml.graph.nn.encoder.mean.EdgeMean",
                },
                "predictor": {
                    "in_dim": len(cbb_features.GRAPH_STATS_COLUMNS),
                    "hidden_dim": 300,
                    "out_dim": 1,
                },
            },
            "cbb.pyg.fit.trainer": {
                "devices": 1,
                "max_epochs": 100,
                "logger": MLFlowLogger("cbb", tracking_uri="sqlite:///mlflow.db"),
                "callbacks": [
                    EarlyStopping(monitor="val_loss", patience=50, mode="min"),
                    ModelCheckpoint(monitor="val_loss", mode="min"),
                ]
            },
            "cbb.sklearn.fit.games": "data/cbb/raw.csv",
            "cbb.sklearn.fit.model": {
                "class_path": "sklearn.ensemble.RandomForestRegressor"
            },
            "cbb.sklearn.fit.stats_columns": cbb_features.STATS_COLUMNS,
            "cbb.sklearn.fit.categorical_columns": cbb_features.CATEGORICAL_COLUMNS,
            "cbb.sklearn.fit.target_column": cbb_features.TARGET_COLUMN,
            "cbb.sklearn.fit.season_column": cbb_features.SEASON_COLUMN,
            "cbb.sklearn.fit.date_column": cbb_features.DATE_COLUMN,
            "cbb.sklearn.fit.team_column": cbb_features.TEAM_COLUMN,
            "cbb.sklearn.fit.team_opp_column": cbb_features.TEAM_OPP_COLUMN,
            "cbb.sklearn.fit.print_metrics": True,
            "cfb.download.output_file": "data/cfb/raw.csv",
            "cfb.process.games": "data/cfb/raw.csv",
            "cfb.process.stats_columns": cfb_features.GRAPH_FEATURES,
            "cfb.process.target_column": cfb_features.TARGET_COLUMN,
            "cfb.process.output_file": "data/cfb/data.csv",
            "cfb.sklearn.fit.games": "data/cfb/data.csv",
            "cfb.sklearn.fit.model": {
                "class_path": "sklearn.ensemble.RandomForestRegressor"
            },
            "cfb.sklearn.fit.stats_columns": cfb_features.STATS_COLUMNS,
            "cfb.sklearn.fit.categorical_columns": cfb_features.CATEGORICAL_COLUMNS,
            "cfb.sklearn.fit.target_column": cfb_features.TARGET_COLUMN,
            "cfb.sklearn.fit.season_column": cfb_features.SEASON_COLUMN,
            "cfb.sklearn.fit.date_column": cfb_features.DATE_COLUMN,
            "cfb.sklearn.fit.team_column": cfb_features.TEAM_COLUMN,
            "cfb.sklearn.fit.team_opp_column": cfb_features.TEAM_OPP_COLUMN,
            "cfb.sklearn.fit.print_metrics": True,
            "nba.download.output_file": "data/nba/raw.csv",
            "nba.process.games": "data/nba/raw.csv",
            "nba.process.stats_columns": nba_features.GRAPH_FEATURES,
            "nba.process.target_column": nba_features.TARGET_COLUMN,
            "nba.process.output_file": "data/nba/data.csv",
            "nba.sklearn.fit.games": "data/nba/data.csv",
            "nba.sklearn.fit.model": {
                "class_path": "sklearn.ensemble.RandomForestRegressor"
            },
            "nba.sklearn.fit.stats_columns": nba_features.STATS_COLUMNS,
            "nba.sklearn.fit.categorical_columns": nba_features.CATEGORICAL_COLUMNS,
            "nba.sklearn.fit.target_column": nba_features.TARGET_COLUMN,
            "nba.sklearn.fit.season_column": nba_features.SEASON_COLUMN,
            "nba.sklearn.fit.date_column": nba_features.DATE_COLUMN,
            "nba.sklearn.fit.team_column": nba_features.TEAM_COLUMN,
            "nba.sklearn.fit.team_opp_column": nba_features.TEAM_OPP_COLUMN,
            "nba.sklearn.fit.print_metrics": True,
            "nfl.download.output_file": "data/nfl/raw.csv",
            "nfl.process.games": "data/nfl/raw.csv",
            "nfl.process.stats_columns": nfl_features.GRAPH_FEATURES,
            "nfl.process.target_column": nfl_features.TARGET_COLUMN,
            "nfl.process.output_file": "data/nfl/data.csv",
            "nfl.sklearn.fit.games": "data/nfl/data.csv",
            "nfl.sklearn.fit.model": {
                "class_path": "sklearn.ensemble.RandomForestRegressor"
            },
            "nfl.sklearn.fit.stats_columns": nfl_features.STATS_COLUMNS,
            "nfl.sklearn.fit.meta_columns": nfl_features.META_COLUMNS,
            "nfl.sklearn.fit.categorical_columns": nfl_features.CATEGORICAL_COLUMNS,
            "nfl.sklearn.fit.target_column": nfl_features.TARGET_COLUMN,
            "nfl.sklearn.fit.season_column": nfl_features.SEASON_COLUMN,
            "nfl.sklearn.fit.date_column": nfl_features.DATE_COLUMN,
            "nfl.sklearn.fit.team_column": nfl_features.TEAM_COLUMN,
            "nfl.sklearn.fit.team_opp_column": nfl_features.TEAM_OPP_COLUMN,
            "nfl.sklearn.fit.print_metrics": True,
        },
        parser_mode="omegaconf+",
    )


if __name__ == "__main__":
    cli()
