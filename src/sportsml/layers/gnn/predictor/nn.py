from typing import List

import torch


class NNPredictor(torch.nn.Module):
    def __init__(
        self,
        in_feats: int,
        dims: List = [],
        out_feats: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dims = list(dims)
        self.dims.insert(0, in_feats)
        layers = [torch.nn.BatchNorm1d(in_feats)]
        for idx in range(len(self.dims) - 1):
            layers.append(torch.nn.Linear(self.dims[idx], self.dims[idx + 1]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.BatchNorm1d(self.dims[idx + 1]))
        layers.append(torch.nn.Linear(self.dims[-1], out_feats))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, g, h, e):
        g = g.local_var()
        src, dst = g.edges()
        x = torch.cat([h[src], h[dst], e], dim=1)
        return self.mlp(x)
