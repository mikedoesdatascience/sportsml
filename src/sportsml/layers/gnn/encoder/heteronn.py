import dgl
import torch
from lightning.pytorch.core.mixins import HyperparametersMixin


class HeteroNNEncoder(torch.nn.Module, HyperparametersMixin):
    def __init__(self, in_feats, out_feats=100, dropout=0.1, init=True, readout=True):
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__
        self.init = None
        self.readout = None
        if init:
            self.init = True
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
        if readout:
            readout_in_feats = out_feats if self.init else in_feats
            self.readout = torch.nn.Sequential(
                torch.nn.BatchNorm1d(readout_in_feats),
                torch.nn.Linear(readout_in_feats, out_feats),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            )

    def edge_to_node(self, g):
        g.multi_update_all(
            {
                "win": (
                    dgl.function.copy_e("h", "m"),
                    dgl.function.reducer.mean("m", "h"),
                ),
                "loss": (
                    dgl.function.copy_e("h", "m"),
                    dgl.function.reducer.mean("m", "h"),
                ),
            },
            "mean",
        )

    def forward(self, g):
        g = g.local_var()

        if self.init is not None:
            g.edges["win"].data["h"] = self.w_init(g.edges["win"].data["f"])
            g.edges["loss"].data["h"] = self.l_init(g.edges["loss"].data["f"])
        else:
            g.edges["win"].data["h"] = g.edges["win"].data["f"]
            g.edges["loss"].data["h"] = g.edges["loss"].data["f"]

        self.edge_to_node(g)

        if self.readout is not None:
            g.ndata["h"] = self.readout(g.ndata["h"])
        
        return g.ndata["h"]

        
