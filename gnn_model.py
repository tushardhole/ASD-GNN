import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class FCGraphGNN(nn.Module):
    """
    Simple 2-layer GCN for FC matrices
    """
    def __init__(self, in_feats=5, hidden=64, out_feats=2):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.fc_out = nn.Linear(hidden, out_feats)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, getattr(data, 'edge_attr', None), getattr(data, 'batch', None)
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_attr.squeeze() if edge_attr is not None else None))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_attr.squeeze() if edge_attr is not None else None))
        x = global_mean_pool(x, batch) if batch is not None else x.mean(dim=0, keepdim=True)
        x = self.fc_out(x)
        return x
