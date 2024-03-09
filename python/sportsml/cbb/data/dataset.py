from .features import GRAPH_FEATURES
from ...utils.dataset import HeteroGraphDataset


class CBBHeteroGraphDataset(HeteroGraphDataset):
    def __init__(self, games):
        super().__init__(
            games=games,
            feature_columns=GRAPH_FEATURES,
            target_columns=["PlusMinus"],
            win_column="Won",
            home_column="Loc",
            season_column="Season",
            date_column="DayNum",
            team_column="Team",
            num_nodes=378,
        )
