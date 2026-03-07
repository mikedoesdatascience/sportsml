import torch
from lightning.pytorch.core.mixins import HyperparametersMixin


class EdgeFFN(torch.nn.Module, HyperparametersMixin):
    """Edge predictor that concatenates source and target node features
    and projects them into `out_dim`.

    Args:
            in_dim: dimensionality of input node features
            out_dim: dimensionality of output edge features / predictions
            hidden_dim: optional hidden dimension for a single hidden layer
            activation: activation module class or instance (default: ReLU)
    """

    def __init__(
        self, in_dim: int, out_dim: int, hidden_dim: int | None = None, activation=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        if activation is None:
            activation = torch.nn.ReLU()
        # when concatenating source and target, input becomes 2 * in_dim
        proj_in = in_dim * 2
        if hidden_dim is None:
            self.net = torch.nn.Linear(proj_in, out_dim)
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(proj_in, hidden_dim),
                activation,
                torch.nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute edge predictions from node features.

        Args:
                x: node feature tensor of shape [num_nodes, in_dim]
                edge_index: long tensor of shape [2, num_edges] with source,dest indices

        Returns:
                Tensor of shape [num_edges, out_dim]
        """
        if x is None:
            raise ValueError("node features `x` must be provided")
        if edge_index is None:
            raise ValueError("`edge_index` must be provided")
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        src = x[src_idx]
        dst = x[dst_idx]
        # concat and project
        h = torch.cat([src, dst], dim=-1)
        return self.net(h)
