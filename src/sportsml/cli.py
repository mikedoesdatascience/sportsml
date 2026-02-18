import jsonargparse

from . import __version__
from .cfb.data.download import download as cfb_download
from .nba.data.download import download as nba_download
from .nfl.data.download import download as nfl_download


def version():
    """Print the version of the sportsml package."""
    print(__version__)


def cli():
    jsonargparse.auto_cli(
        {
            "cfb": {"download": cfb_download},
            "nba": {"download": nba_download},
            "nfl": {"download": nfl_download},
            "version": version,
        },
        as_positional=False,
        set_defaults={
            "cfb.download.output_file": "data/cfb/raw.csv",
            "nba.download.output_file": "data/nba/raw.csv",
            "nfl.download.output_file": "data/nfl/raw.csv",
        },
        parser_mode="omegaconf+"
    )


if __name__ == "__main__":
    cli()
