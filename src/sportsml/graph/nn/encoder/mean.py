import torch
from lightning.pytorch.core.mixins import HyperparametersMixin
from torch_geometric.nn import MessagePassing


class EdgeMean(MessagePassing, HyperparametersMixin):
    """
    Aggregates edge features into node features using mean aggregation.
    If node_in_channels is provided, the (linearly projected) node features
    are added to the aggregated edge features.
    """

    def __init__(
        self,
        in_edge_channels: int,
    ):
        super().__init__(aggr="mean")
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__

        self.bn = torch.nn.BatchNorm1d(in_edge_channels)

    def forward(self, edge_attr, edge_index, **kwargs):
        """
        Args:
            edge_attr: Tensor with shape [num_edges, edge_feature_dim]
            edge_index: LongTensor with shape [2, num_edges]
        Returns:
            Tensor of shape [num_nodes, edge_feature_dim]: node features produced from aggregated edges
        """
        if edge_attr is None:
            raise ValueError("edge_attr must be provided")
        aggr_edges = self.propagate(
            edge_index, edge_attr=edge_attr
        )  # [num_nodes, edge_feature_dim]
        aggr_edges = self.bn(aggr_edges)
        return aggr_edges

    def message(self, edge_attr):
        return edge_attr

    def update(self, aggr_out):
        return aggr_out
