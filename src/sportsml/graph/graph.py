import pandas as pd
import torch
import torch_geometric as pyg


def create_graph(
    games: pd.DataFrame,
    stats_columns: list[str],
    target_column: str,
    season_column: str = "season",
    date_column: str = "date",
):
    num_nodes = games["src"].max() + 1
    edge_index = torch.tensor([games["src"].tolist(), games["dst"].tolist()])
    y = torch.tensor(games[[target_column]].values, dtype=torch.float)
    edge_attr = torch.tensor(games[stats_columns].values, dtype=torch.float)
    season = torch.tensor(games[season_column].values, dtype=torch.float)
    date = torch.tensor(games[date_column].values, dtype=torch.float)

    graph = pyg.data.Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
        y=y,
        edge_attr=edge_attr,
        season=season,
        date=date,
    )

    return graph


def create_complete_graph(num_nodes: int):
    """
    Generates a complete, undirected graph as a PyG Data object.

    Args:
        num_nodes (int): The number of nodes in the graph.

    Returns:
        torch_geometric.data.Data: The PyG Data object for the complete graph.
    """
    
    # 1. Generate all possible source and target pairs (excluding self-loops)
    # The combination of all possible pairs forms a directed complete graph.
    # The code below effectively generates two arrays, one for source nodes, one for target nodes.
    source_nodes = torch.arange(num_nodes).repeat_interleave(num_nodes)
    target_nodes = torch.arange(num_nodes).repeat(num_nodes)
    
    # Filter out self-loops (edges from a node to itself)
    # A complete graph is typically defined without self-loops unless specified otherwise.
    mask = source_nodes != target_nodes
    source_nodes = source_nodes[mask]
    target_nodes = target_nodes[mask]
    
    # Combine into edge_index in COO format (shape [2, num_edges])
    edge_index = torch.stack([source_nodes, target_nodes], dim=0).long()

    # 3. Create the PyTorch Geometric Data object
    data = pyg.data.Data(edge_index=edge_index)
    
    # Ensure num_nodes is explicitly set to avoid potential issues with isolated nodes in batching
    # if the feature matrix 'x' is not used
    data.num_nodes = num_nodes 
    
    return data