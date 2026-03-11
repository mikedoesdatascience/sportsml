import torch
from lightning.pytorch.core.mixins import HyperparametersMixin
from torch_geometric.nn import MessagePassing


class EdgeEncoder(MessagePassing, HyperparametersMixin):
    """
    Aggregates edge features into node features using mean aggregation.
    If node_in_channels is provided, the (linearly projected) node features
    are added to the aggregated edge features.
    """

    def __init__(
        self,
        in_edge_channels: int,
        out_channels: int,
        node_in_channels: int | None = None,
    ):
        super().__init__(aggr="mean")
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__

        self.bn = torch.nn.BatchNorm1d(in_edge_channels)
        self.lin_edge = torch.nn.Linear(in_edge_channels, out_channels)
        self.node_in_channels = node_in_channels
        if node_in_channels is not None:
            self.lin_node = torch.nn.Linear(node_in_channels, out_channels)
        else:
            self.lin_node = None
        self.act = torch.nn.ReLU()

    def forward(self, edge_attr, edge_index, x=None, meta_attr=None, cat_attr=None):
        """
        Args:
            edge_attr: Tensor with shape [num_edges, in_edge_channels]
            edge_index: LongTensor with shape [2, num_edges]
            x: optional node features Tensor with shape [num_nodes, node_in_channels]
        Returns:
            Tensor of shape [num_nodes, out_channels]: node features
            produced from aggregated edges
        """
        if edge_attr is None:
            raise ValueError("edge_attr must be provided")
        edge_attr = self.bn(edge_attr)
        edge_attr = self.lin_edge(edge_attr)
        aggr = self.propagate(edge_index, edge_attr=edge_attr)
        aggr = self.act(aggr)

        if self.lin_node is not None and x is not None:
            node_proj = self.lin_node(x)
            return aggr + node_proj
        return aggr

    def message(self, edge_attr):
        # lin already applied in forward
        return edge_attr

    def update(self, aggr_out):
        # identity here; activation handled in forward
        return aggr_out
