import dgl
import torch
from lightning.pytorch.core.mixins import HyperparametersMixin

from ...mlp import MLP


class HeteroNNPredictor(torch.nn.Module, HyperparametersMixin):
    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        depth: int,
        out_feats: int = 1,
        dropout: float = 0.1,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__
        self.home_src_lin = MLP(
            in_feats=in_feats,
            hidden_dim=hidden_dim,
            depth=depth,
            out_feats=out_feats,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        self.home_dst_lin = MLP(
            in_feats=in_feats,
            hidden_dim=hidden_dim,
            depth=depth,
            out_feats=out_feats,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        self.away_src_lin = MLP(
            in_feats=in_feats,
            hidden_dim=hidden_dim,
            depth=depth,
            out_feats=out_feats,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        self.away_dst_lin = MLP(
            in_feats=in_feats,
            hidden_dim=hidden_dim,
            depth=depth,
            out_feats=out_feats,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    def forward(self, g):
        g = g.local_var()
        g.ndata["hu"] = self.home_src_lin(g.ndata["h"])
        g.ndata["hv"] = self.home_dst_lin(g.ndata["h"])
        g.ndata["au"] = self.away_src_lin(g.ndata["h"])
        g.ndata["av"] = self.away_dst_lin(g.ndata["h"])
        g.apply_edges(dgl.function.u_add_v("hu", "hv", "hp"))
        g.apply_edges(dgl.function.u_add_v("au", "av", "ap"))
        return g.edges["home"].data["hp"], g.edges["away"].data["ap"]
