import jsonargparse

from .cfb.data.download import download


def cli():
    jsonargparse.auto_cli(
        {
            "cfb": {
                "download": download
            }
        },
        as_positional=False,
    )

if __name__ == "__main__":
    cli()