import dgl
import torch


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_feats, out_feats=100, depth=3, dropout=0.1, use_weight=False):
        super().__init__()
        self.use_weight = use_weight
        self.layer_init = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_feats),
            torch.nn.Linear(in_feats, out_feats),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.BatchNorm1d(3 * out_feats + 1),
                    torch.nn.Linear(3 * out_feats + 1, out_feats),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                )
                for _ in range(depth)
            ]
        )
        self.layer_out = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * out_feats),
            torch.nn.Linear(2 * out_feats, out_feats),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, g, e):
        g = g.local_var()
        g.edata["f"] = self.layer_init(e)
        if self.use_weight:
            g.edata["f"] = g.edata["f"] * g.edata["w"]
        g.update_all(dgl.function.copy_e("f", "m"), dgl.function.reducer.mean("m", "f"))

        g.ndata["h"] = g.ndata["f"].clone()

        src, dst = g.edges()
        for layer in self.layers:
            x = torch.cat(
                [
                    g.ndata["h"][src],
                    g.edata["f"],
                    g.edata["p"],
                    g.ndata["h"][dst],
                ],
                dim=1,
            )
            x = layer(x)
            g.edata["h"] = x
            g.update_all(
                dgl.function.copy_e("h", "m"),
                dgl.function.reducer.mean("m", "h"),
            )
        return self.layer_out(torch.cat([g.ndata["f"], g.ndata["h"]], dim=1))
