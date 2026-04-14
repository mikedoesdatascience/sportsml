import pathlib

import pandas as pd


def process(
    games: pd.DataFrame,
    stats_columns: list[str],
    target_column: str,
    output_file: str = None,
):
    games = games.dropna(subset=stats_columns + [target_column])

    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    if output_file:
        games.to_csv(output_file, index=False)

    return games
