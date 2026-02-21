import jsonargparse
import jsonargparse.typing
import pandas as pd

from . import __version__
from .cbb.data.download import download as cbb_download
from .cbb.data.features import CATEGORICAL_COLUMNS as CBB_CATEGORICAL_COLUMNS
from .cbb.data.features import STATS_COLUMNS as CBB_STATS_COLUMNS
from .cfb.data.download import download as cfb_download
from .cfb.data.features import CATEGORICAL_COLUMNS as CFB_CATEGORICAL_COLUMNS
from .cfb.data.features import STATS_COLUMNS as CFB_STATS_COLUMNS
from .models.rf import train_rf
from .nba.data.download import download as nba_download
from .nfl.data.download import download as nfl_download
from .nfl.data.features import CATEGORICAL_COLUMNS as NFL_CATEGORICAL_COLUMNS
from .nfl.data.features import META_COLUMNS as NFL_META_COLUMNS
from .nfl.data.features import STATS_COLUMNS as NFL_STATS_COLUMNS

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
            "cbb.rf.train.stats_columns": CBB_STATS_COLUMNS,
            "cbb.rf.train.categorical_columns": CBB_CATEGORICAL_COLUMNS,
            "cbb.rf.train.target_column": "PlusMinus",
            "cbb.rf.train.season_column": "Season",
            "cbb.rf.train.date_column": "DayNum",
            "cbb.rf.train.team_column": "TeamID",
            "cbb.rf.train.team_opp_column": "TeamID_OPP",
            "cbb.rf.train.print_metrics": True,
            "cfb.download.output_file": "data/cfb/raw.csv",
            "cfb.rf.train.games": "data/cfb/raw.csv",
            "cfb.rf.train.stats_columns": CFB_STATS_COLUMNS,
            "cfb.rf.train.categorical_columns": CFB_CATEGORICAL_COLUMNS,
            "cfb.rf.train.target_column": "result",
            "cfb.rf.train.season_column": "season",
            "cfb.rf.train.date_column": "week",
            "cfb.rf.train.team_column": "team",
            "cfb.rf.train.team_opp_column": "opp_team",
            "cfb.rf.train.print_metrics": True,
            "nba.download.output_file": "data/nba/raw.csv",
            "nfl.download.output_file": "data/nfl/raw.csv",
            "nfl.rf.train.games": "data/nfl/raw.csv",
            "nfl.rf.train.stats_columns": NFL_STATS_COLUMNS,
            "nfl.rf.train.meta_columns": NFL_META_COLUMNS,
            "nfl.rf.train.categorical_columns": NFL_CATEGORICAL_COLUMNS,
            "nfl.rf.train.target_column": "result",
            "nfl.rf.train.season_column": "season",
            "nfl.rf.train.date_column": "week",
            "nfl.rf.train.team_column": "team",
            "nfl.rf.train.team_opp_column": "opp_team",
            "nfl.rf.train.print_metrics": True,
        },
        parser_mode="omegaconf+",
    )


if __name__ == "__main__":
    cli()
