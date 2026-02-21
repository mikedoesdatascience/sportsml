from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeMeanEncoder(MessagePassing):
    """
    Aggregates edge features into node features using mean aggregation.
    If node_in_channels is provided, the (linearly projected) node features
    are added to the aggregated edge features.
    """

    def __init__(self, in_edge_channels: int, out_channels: int, node_in_channels: int | None = None):
        super().__init__(aggr="mean")
        self.lin_edge = nn.Linear(in_edge_channels, out_channels)
        self.node_in_channels = node_in_channels
        if node_in_channels is not None:
            self.lin_node = nn.Linear(node_in_channels, out_channels)
        else:
            self.lin_node = None
        self.act = nn.ReLU()

    def forward(self, edge_index, edge_attr, x=None):
        """
        Args:
            edge_index: LongTensor with shape [2, num_edges]
            edge_attr: Tensor with shape [num_edges, in_edge_channels]
            x: optional node features Tensor with shape [num_nodes, node_in_channels]
        Returns:
            Tensor of shape [num_nodes, out_channels]: node features produced from aggregated edges
        """
        if edge_attr is None:
            raise ValueError("edge_attr must be provided")
        aggr_edges = self.propagate(edge_index, edge_attr=edge_attr)  # [num_nodes, out_channels]
        aggr_edges = self.act(aggr_edges)
        if self.lin_node is not None and x is not None:
            node_proj = self.lin_node(x)
            return aggr_edges + node_proj
        return aggr_edges

    def message(self, edge_attr):
        # transform edge features before aggregation
        return self.lin_edge(edge_attr)

    def update(self, aggr_out):
        # identity here; activation handled in forward
        return aggr_out

