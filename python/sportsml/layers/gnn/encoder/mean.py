import dgl
import torch


class MeanEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g, e):
        g = g.local_var()
        g.edata["f"] = e
        g.update_all(dgl.function.copy_e("f", "m"), dgl.function.reducer.mean("m", "h"))
        return g.ndata["h"]
