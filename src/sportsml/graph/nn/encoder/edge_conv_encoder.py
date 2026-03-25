from __future__ import annotations

from typing import Optional

import torch
from lightning.pytorch.core.mixins import HyperparametersMixin
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj


class EdgeConvEncoderLayer(MessagePassing, HyperparametersMixin):
    """Edge-driven graph convolution layer for sports game graphs.

    Nodes represent teams; edges represent games between teams with box score
    stats as edge attributes. Node representations are learned purely from
    projections of edge attributes, so no initial node features are required.

    Args:
        edge_dim: Dimensionality of input edge attributes (box score stats).
        hidden_dim: Hidden dimensionality of the edge projection MLP.
        out_dim: Output dimensionality of learned node embeddings.
        node_in_dim: Dimensionality of incoming node embeddings from the
            previous layer. None for the first layer (no node features yet).
            When provided the message input is ``concat(edge_attr, x_src)``.
        aggr: Aggregation scheme ('mean', 'sum', or 'max').
        dropout: Dropout probability applied within the edge MLP.
    """

    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int,
        out_dim: int,
        node_in_dim: Optional[int] = None,
        aggr: str = "mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__(aggr=aggr)
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__

        self.node_in_dim = node_in_dim
        mlp_in = edge_dim if node_in_dim is None else edge_dim + node_in_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        edge_index: Adj,
        edge_attr: Tensor,
        x: Optional[Tensor] = None,
    ) -> Tensor:
        """Run one edge-driven message passing step.

        Args:
            edge_index: Graph connectivity of shape [2, E].
            edge_attr: Edge feature matrix of shape [E, edge_dim].
            x: Optional node features of shape [N, *]. Can be None on the
                first layer since no initial node features are assumed.

        Returns:
            Node embedding matrix of shape [N, out_dim].
        """
        n = x.size(0) if x is not None else int(edge_index.max().item()) + 1

        # Build message input: concatenate source-node embeddings from the
        # previous layer with edge attributes so each subsequent layer can
        # attend to a wider neighbourhood.
        if x is not None:
            src = edge_index[0]
            msg_input = torch.cat([edge_attr, x[src]], dim=-1)
        else:
            msg_input = edge_attr

        agg = self.propagate(
            edge_index=edge_index,
            msg_input=msg_input,
            size=(n, n),  # (n, n) ensures correct output shape when x is None
        )
        return self.update_mlp(agg)

    def message(self, msg_input: Tensor) -> Tensor:
        return self.edge_mlp(msg_input)


class EdgeConvEncoder(nn.Module, HyperparametersMixin):
    """Multi-layer edge-driven graph convolution encoder.

    Stacks multiple :class:`EdgeConvEncoderLayer` layers to build progressively
    richer team node embeddings from game box score edge attributes.

    Args:
        edge_dim: Dimensionality of input edge attributes (box score stats).
        hidden_dim: Hidden dimensionality used in intermediate layers.
        out_dim: Output dimensionality of the final node embeddings.
        num_layers: Number of message passing layers (>= 1).
        aggr: Aggregation scheme ('mean', 'sum', or 'max').
        dropout: Dropout probability applied within edge MLPs.
    """

    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        aggr: str = "mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__

        layers = []
        for i in range(num_layers):
            # First layer has no prior node embeddings; every subsequent layer
            # receives the hidden_dim output of the layer before it.
            node_in = None if i == 0 else hidden_dim
            layer_out = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(
                EdgeConvEncoderLayer(
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    out_dim=layer_out,
                    node_in_dim=node_in,
                    aggr=aggr,
                    dropout=dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        """Encode all team nodes from game edge attributes.

        Args:
            edge_index: Graph connectivity of shape [2, E].
            edge_attr: Edge feature matrix of shape [E, edge_dim].

        Returns:
            Node embedding matrix of shape [N, out_dim].
        """
        x: Optional[Tensor] = None
        for layer in self.layers:
            x = layer(
                edge_index=edge_index,
                edge_attr=edge_attr,
                x=x,
            )
        return x
