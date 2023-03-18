import dgl
import torch

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_feats, out_feats=100, depth=3):
        super().__init__()
        self.depth = depth
        self.norm = torch.nn.BatchNorm1d(in_feats)
        self.w_i = torch.nn.Linear(in_feats, out_feats)
        self.w_h = torch.nn.Linear(3 * out_feats + 1, out_feats)
    
    def forward(self, g, e):
        g = g.local_var()
        g.edata['f'] = torch.nn.functional.relu(self.w_i(self.norm(e)))
        g.update_all(
            dgl.function.copy_e('f', 'm'),
            dgl.function.reducer.mean('m', 'f')
        )
        
        g.ndata['h'] = g.ndata['f']
        
        src, dst = g.edges()
        for _ in range(self.depth):
            x = torch.cat([g.ndata['h'][src], g.edata['f'], g.edata['p'], g.ndata['h'][dst]], dim=1)
            x = torch.nn.functional.relu(self.w_h(x))
            g.edata['h'] = x
            g.update_all(
                dgl.function.copy_e('h', 'm'),
                dgl.function.reducer.mean('m', 'h')
            )
        return torch.cat([g.ndata['f'], g.ndata['h']], dim=1)