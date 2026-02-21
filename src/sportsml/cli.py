import jsonargparse
import jsonargparse.typing
import pandas as pd

from . import __version__
from .cbb.data import features as cbb_features
from .cbb.data.download import download as cbb_download
from .cfb.data import features as cfb_features
from .cfb.data.download import download as cfb_download
from .models.sklearn import train_sklearn
from .nba.data import features as nba_features
from .nba.data.download import download as nba_download
from .nfl.data import features as nfl_features
from .nfl.data.download import download as nfl_download

jsonargparse.typing.register_type(
    pd.DataFrame, lambda x: getattr(x, "filename"), lambda x: pd.read_csv(x)
)


def version():
    """Print the version of the sportsml package."""
    print(__version__)


def cli():
    jsonargparse.auto_cli(
        {
            "cbb": {"download": cbb_download, "sklearn": {"train": train_sklearn}},
            "cfb": {"download": cfb_download, "sklearn": {"train": train_sklearn}},
            "nba": {"download": nba_download, "sklearn": {"train": train_sklearn}},
            "nfl": {"download": nfl_download, "sklearn": {"train": train_sklearn}},
            "version": version,
        },
        as_positional=False,
        set_defaults={
            "cbb.download.output_file": "data/cbb/raw.csv",
            "cbb.sklearn.train.games": "data/cbb/raw.csv",
            "cbb.sklearn.train.model": {
                "class_path": "sklearn.ensemble.RandomForestRegressor"
            },
            "cbb.sklearn.train.stats_columns": cbb_features.STATS_COLUMNS,
            "cbb.sklearn.train.categorical_columns": cbb_features.CATEGORICAL_COLUMNS,
            "cbb.sklearn.train.target_column": cbb_features.TARGET_COLUMN,
            "cbb.sklearn.train.season_column": cbb_features.SEASON_COLUMN,
            "cbb.sklearn.train.date_column": cbb_features.DATE_COLUMN,
            "cbb.sklearn.train.team_column": cbb_features.TEAM_COLUMN,
            "cbb.sklearn.train.team_opp_column": cbb_features.TEAM_OPP_COLUMN,
            "cbb.sklearn.train.print_metrics": True,
            "cfb.download.output_file": "data/cfb/raw.csv",
            "cfb.sklearn.train.games": "data/cfb/raw.csv",
            "cfb.sklearn.train.model": {
                "class_path": "sklearn.ensemble.RandomForestRegressor"
            },
            "cfb.sklearn.train.stats_columns": cfb_features.STATS_COLUMNS,
            "cfb.sklearn.train.categorical_columns": cfb_features.CATEGORICAL_COLUMNS,
            "cfb.sklearn.train.target_column": cfb_features.TARGET_COLUMN,
            "cfb.sklearn.train.season_column": cfb_features.SEASON_COLUMN,
            "cfb.sklearn.train.date_column": cfb_features.DATE_COLUMN,
            "cfb.sklearn.train.team_column": cfb_features.TEAM_COLUMN,
            "cfb.sklearn.train.team_opp_column": cfb_features.TEAM_OPP_COLUMN,
            "cfb.sklearn.train.print_metrics": True,
            "nba.download.output_file": "data/nba/raw.csv",
            "nba.sklearn.train.games": "data/nba/raw.csv",
            "nba.sklearn.train.model": {
                "class_path": "sklearn.ensemble.RandomForestRegressor"
            },
            "nba.sklearn.train.stats_columns": nba_features.STATS_COLUMNS,
            "nba.sklearn.train.categorical_columns": nba_features.CATEGORICAL_COLUMNS,
            "nba.sklearn.train.target_column": nba_features.TARGET_COLUMN,
            "nba.sklearn.train.season_column": nba_features.SEASON_COLUMN,
            "nba.sklearn.train.date_column": nba_features.DATE_COLUMN,
            "nba.sklearn.train.team_column": nba_features.TEAM_COLUMN,
            "nba.sklearn.train.team_opp_column": nba_features.TEAM_OPP_COLUMN,
            "nba.sklearn.train.print_metrics": True,
            "nfl.download.output_file": "data/nfl/raw.csv",
            "nfl.sklearn.train.games": "data/nfl/raw.csv",
            "nfl.sklearn.train.model": {
                "class_path": "sklearn.ensemble.RandomForestRegressor"
            },
            "nfl.sklearn.train.stats_columns": nfl_features.STATS_COLUMNS,
            "nfl.sklearn.train.meta_columns": nfl_features.META_COLUMNS,
            "nfl.sklearn.train.categorical_columns": nfl_features.CATEGORICAL_COLUMNS,
            "nfl.sklearn.train.target_column": nfl_features.TARGET_COLUMN,
            "nfl.sklearn.train.season_column": nfl_features.SEASON_COLUMN,
            "nfl.sklearn.train.date_column": nfl_features.DATE_COLUMN,
            "nfl.sklearn.train.team_column": nfl_features.TEAM_COLUMN,
            "nfl.sklearn.train.team_opp_column": nfl_features.TEAM_OPP_COLUMN,
            "nfl.sklearn.train.print_metrics": True,
        },
        parser_mode="omegaconf+",
    )


if __name__ == "__main__":
    cli()
