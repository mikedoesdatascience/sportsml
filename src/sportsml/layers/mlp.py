import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_feats,
        hidden_dim,
        depth,
        out_feats=1,
        dropout=0.1,
        batch_norm=True,
    ):
        super().__init__()
        dims = [in_feats] + [hidden_dim] * (depth)
        layers = [torch.nn.BatchNorm1d(in_feats)] if batch_norm else []
        for idx in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.ReLU())
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(dims[idx + 1]))
        layers.append(torch.nn.Linear(dims[-1], out_feats))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
