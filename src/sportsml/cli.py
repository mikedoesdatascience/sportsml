import jsonargparse

from .cfb.data.download import download as cfb_download
from .nba.data.download import download as nba_download
from .nfl.data.download import download as nfl_download


def cli():
    jsonargparse.auto_cli(
        {
            "cfb": {
                "download": cfb_download
            },
            "nba": {
                "download": nba_download
            },
            "nfl": {
                "download": nfl_download
            }
        },
        as_positional=False,
    )

if __name__ == "__main__":
    cli()