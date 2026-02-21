import jsonargparse
import jsonargparse.typing
import pandas as pd

from . import __version__
from .cbb.data import features as cbb_features
from .cbb.data.download import download as cbb_download
from .cfb.data import features as cfb_features
from .cfb.data.download import download as cfb_download
from .models.rf import train_rf
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
            "cbb": {"download": cbb_download, "rf": {"train": train_rf}},
            "cfb": {"download": cfb_download, "rf": {"train": train_rf}},
            "nba": {"download": nba_download, "rf": {"train": train_rf}},
            "nfl": {"download": nfl_download, "rf": {"train": train_rf}},
            "version": version,
        },
        as_positional=False,
        set_defaults={
            "cbb.download.output_file": "data/cbb/raw.csv",
            "cbb.rf.train.games": "data/cbb/raw.csv",
            "cbb.rf.train.stats_columns": cbb_features.STATS_COLUMNS,
            "cbb.rf.train.categorical_columns": cbb_features.CATEGORICAL_COLUMNS,
            "cbb.rf.train.target_column": cbb_features.TARGET_COLUMN,
            "cbb.rf.train.season_column": cbb_features.SEASON_COLUMN,
            "cbb.rf.train.date_column": cbb_features.DATE_COLUMN,
            "cbb.rf.train.team_column": cbb_features.TEAM_COLUMN,
            "cbb.rf.train.team_opp_column": cbb_features.TEAM_OPP_COLUMN,
            "cbb.rf.train.print_metrics": True,
            "cfb.download.output_file": "data/cfb/raw.csv",
            "cfb.rf.train.games": "data/cfb/raw.csv",
            "cfb.rf.train.stats_columns": cfb_features.STATS_COLUMNS,
            "cfb.rf.train.categorical_columns": cfb_features.CATEGORICAL_COLUMNS,
            "cfb.rf.train.target_column": cfb_features.TARGET_COLUMN,
            "cfb.rf.train.season_column": cfb_features.SEASON_COLUMN,
            "cfb.rf.train.date_column": cfb_features.DATE_COLUMN,
            "cfb.rf.train.team_column": cfb_features.TEAM_COLUMN,
            "cfb.rf.train.team_opp_column": cfb_features.TEAM_OPP_COLUMN,
            "cfb.rf.train.print_metrics": True,
            "nba.download.output_file": "data/nba/raw.csv",
            "nfl.download.output_file": "data/nfl/raw.csv",
            "nfl.rf.train.games": "data/nfl/raw.csv",
            "nfl.rf.train.stats_columns": nfl_features.STATS_COLUMNS,
            "nfl.rf.train.meta_columns": nfl_features.META_COLUMNS,
            "nfl.rf.train.categorical_columns": nfl_features.CATEGORICAL_COLUMNS,
            "nfl.rf.train.target_column": nfl_features.TARGET_COLUMN,
            "nfl.rf.train.season_column": nfl_features.SEASON_COLUMN,
            "nfl.rf.train.date_column": nfl_features.DATE_COLUMN,
            "nfl.rf.train.team_column": nfl_features.TEAM_COLUMN,
            "nfl.rf.train.team_opp_column": nfl_features.TEAM_OPP_COLUMN,
            "nfl.rf.train.print_metrics": True,
        },
        parser_mode="omegaconf+",
    )


if __name__ == "__main__":
    cli()
