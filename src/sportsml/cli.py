import jsonargparse
import jsonargparse.typing
import pandas as pd

from . import __version__
from .cbb.data.download import download as cbb_download
from .cbb.data.features import STATS_COLUMNS as CBB_STATS_COLUMNS
from .cfb.data.download import download as cfb_download
from .models.rf import train_rf
from .nba.data.download import download as nba_download
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
            "cfb": {"download": cfb_download},
            "nba": {"download": nba_download},
            "nfl": {"download": nfl_download},
            "version": version,
        },
        as_positional=False,
        set_defaults={
            "cbb.download.output_file": "data/cbb/raw.csv",
            "cbb.rf.train.stats_columns": CBB_STATS_COLUMNS,
            "cbb.rf.train.target_column": "PlusMinus",
            "cbb.rf.train.season_column": "Season",
            "cbb.rf.train.date_column": "DayNum",
            "cbb.rf.train.team_column": "TeamID",
            "cbb.rf.train.team_opp_column": "TeamID_OPP",
            "cbb.rf.train.print_metrics": True,
            "cfb.download.output_file": "data/cfb/raw.csv",
            "nba.download.output_file": "data/nba/raw.csv",
            "nfl.download.output_file": "data/nfl/raw.csv",
        },
        parser_mode="omegaconf+",
    )


if __name__ == "__main__":
    cli()
