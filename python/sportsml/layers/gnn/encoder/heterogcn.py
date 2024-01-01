import dgl
import torch
from lightning.pytorch.core.mixins import HyperparametersMixin


class HeteroGCNEncoder(torch.nn.Module, HyperparametersMixin):
    def __init__(self, in_feats, out_feats=100, depth=3, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__
        self.depth = depth
        self.w_init = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_feats),
            torch.nn.Linear(in_feats, out_feats),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.l_init = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_feats),
            torch.nn.Linear(in_feats, out_feats),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.src_layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(out_feats),
            torch.nn.Linear(out_feats, out_feats),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.dst_layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(out_feats),
            torch.nn.Linear(out_feats, out_feats),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.w_layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(out_feats),
            torch.nn.Linear(out_feats, out_feats),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.l_layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(out_feats),
            torch.nn.Linear(out_feats, out_feats),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.layer_out = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * out_feats),
            torch.nn.Linear(2 * out_feats, out_feats),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

    def edge_to_node(self, g):
        g.multi_update_all(
            {
                "won": (
                    dgl.function.copy_e("h", "m"),
                    dgl.function.reducer.mean("m", "h"),
                ),
                "lost": (
                    dgl.function.copy_e("h", "m"),
                    dgl.function.reducer.mean("m", "h"),
                ),
            },
            "mean",
        )

    def forward(self, g):
        g = g.local_var()

        g.edges["won"].data["h"] = self.w_init(g.edges["won"].data["f"])
        g.edges["lost"].data["h"] = self.l_init(g.edges["lost"].data["f"])

        self.edge_to_node(g)

        g.ndata["f"] = g.ndata["h"].clone()

        for _ in range(self.depth):
            g.ndata["hu"] = self.src_layer(g.ndata["h"])
            g.ndata["hv"] = self.dst_layer(g.ndata["h"])
            g.apply_edges(dgl.function.u_add_v("hu", "hv", "h"))

            g.edges["won"].data["h"] = self.w_layer(g.edges["won"].data["h"])
            g.edges["lost"].data["h"] = self.w_layer(g.edges["lost"].data["h"])

            self.edge_to_node(g)

        return self.layer_out(torch.cat([g.ndata["f"], g.ndata["h"]], dim=1))
