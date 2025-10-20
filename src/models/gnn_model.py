import torch, torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

class EdgeReadout(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim*2, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x, edge_index):
        src, dst = edge_index
        pair = torch.cat([x[src], x[dst]], dim=-1)
        return self.mlp(pair)

class SimpleGCN(nn.Module):
    def __init__(self, in_features=1, hidden=32, out_features=16):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden)
        self.conv2 = GCNConv(hidden, out_features)
        self.readout = EdgeReadout(out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        logits = self.readout(x, edge_index)
        return F.log_softmax(logits, dim=1)
