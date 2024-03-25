from .features import STATS_COLUMNS
from ...utils.dataset import HeteroGraphDataset


class CBBHeteroGraphDataset(HeteroGraphDataset):
    def __init__(self, games):
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
        )
