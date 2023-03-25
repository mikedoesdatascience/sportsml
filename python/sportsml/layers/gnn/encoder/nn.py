import dgl
import torch


class NNEncoder(torch.nn.Module):
    def __init__(self, in_feats, out_feats=100):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(in_feats)
        self.w = torch.nn.Linear(in_feats, out_feats)

    def forward(self, g, e):
        g = g.local_var()
        g.edata["f"] = torch.nn.functional.relu(self.w(self.norm(e)))
        g.update_all(dgl.function.copy_e("f", "m"), dgl.function.reducer.mean("m", "h"))
        return g.ndata["h"]
