from typing import List

import pandas as pd

from .features import STATS_COLUMNS
from ...utils.datamodule import HeteroGraphDataModule


class CBBHeteroGraphDataModule(HeteroGraphDataModule):
    num_feats: int = len(STATS_COLUMNS)
    num_targets: int = 1

    def __init__(
        self,
        games: pd.DataFrame,
        batch_size: int = 4,
        split_type: str = "random",
        splits: List[int] = [0.8, 0.1, 0.1],
        num_workers: int = 4,
    ):
        super().__init__(
            games=games,
            feature_columns=STATS_COLUMNS,
            target_columns=["PlusMinus"],
            win_column="Win",
            loc_column="Loc",
            season_column="Season",
            date_column="DayNum",
            team_column="TeamID",
            num_nodes=378,
            batch_size=batch_size,
            split_type=split_type,
            splits=splits,
            num_workers=num_workers,
        )
