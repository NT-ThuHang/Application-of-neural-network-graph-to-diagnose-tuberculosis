import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU

from torch_geometric.nn import GINConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool


def MLP(input_channels, hidden_channels):
    return nn.Sequential(Linear(input_channels, hidden_channels),
                         ReLU(), 
                         Linear(hidden_channels, hidden_channels))

class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(GCN, self).__init__()
        self.GIN = GINConv(MLP(input_channels, hidden_channels))
        self.GIN1 = GINConv(MLP(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index, batch):
        x = self.GIN(x, edge_index).relu()
        x = self.GIN1(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        return x
