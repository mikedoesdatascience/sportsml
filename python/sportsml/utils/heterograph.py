import dgl
import torch


def heterograph_encoder(src, dst, feat, won_mask, num_nodes):
    graph = dgl.heterograph(
        {
            ("team", "won", "team"): (
                (torch.tensor(src[won_mask]), torch.tensor(dst[won_mask]))
            ),
            ("team", "lost", "team"): (
                (torch.tensor(src[~won_mask]), torch.tensor(dst[~won_mask]))
            ),
        },
        {"team": num_nodes},
    )
    graph.edges["won"].data["f"] = torch.tensor(feat[won_mask]).float()
    graph.edges["lost"].data["f"] = torch.tensor(feat[~won_mask]).float()
    return graph


def heterograph_predictor(src, dst, feat, home_mask, num_nodes, y=None):
    graph = dgl.heterograph(
        {
            ("team", "home", "team"): (
                (torch.tensor(src[home_mask]), torch.tensor(dst[home_mask]))
            ),
            ("team", "away", "team"): (
                (torch.tensor(src[~home_mask]), torch.tensor(dst[~home_mask]))
            ),
        },
        {"team": num_nodes},
    )
    graph.edges["home"].data["f"] = torch.tensor(feat[home_mask]).float()
    graph.edges["away"].data["f"] = torch.tensor(feat[~home_mask]).float()
    if y is not None:
        graph.edges["home"].data["y"] = torch.tensor(y[home_mask]).float()
        graph.edges["away"].data["y"] = torch.tensor(y[~home_mask]).float()
    return graph
