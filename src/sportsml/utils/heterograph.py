import dgl
import torch


def heterograph_encoder(src, dst, feat, win_mask, num_nodes):
    win_mask = win_mask == 1
    graph = dgl.heterograph(
        {
            ("team", "win", "team"): (
                (torch.tensor(src[win_mask]), torch.tensor(dst[win_mask]))
            ),
            ("team", "loss", "team"): (
                (torch.tensor(src[~win_mask]), torch.tensor(dst[~win_mask]))
            ),
        },
        {"team": num_nodes},
    )
    graph.edges["win"].data["f"] = torch.tensor(feat[win_mask]).float()
    graph.edges["loss"].data["f"] = torch.tensor(feat[~win_mask]).float()
    return graph


def heterograph_predictor(src, dst, feat, loc, num_nodes, y=None):
    home_mask = loc == 1
    graph = dgl.heterograph(
        {
            ("team", "home", "team"): (
                (
                    torch.tensor(src[home_mask]),
                    torch.tensor(dst[home_mask]),
                )
            ),
            ("team", "away", "team"): (
                (
                    torch.tensor(src[~home_mask]),
                    torch.tensor(dst[~home_mask]),
                )
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
