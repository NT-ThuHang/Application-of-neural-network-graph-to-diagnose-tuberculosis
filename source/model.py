import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU

from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool


def MLP(input_channels, hidden_channels):
    return nn.Sequential(Linear(input_channels, hidden_channels),
                         ReLU(), 
                         Linear(hidden_channels, hidden_channels))

class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.GIN = GINConv(MLP(input_channels, hidden_channels))
        self.GIN1 = GINConv(MLP(hidden_channels, hidden_channels))
        self.GIN2 = GINConv(MLP(hidden_channels, hidden_channels))

        self.GCN = GCNConv(input_channels, hidden_channels)
        self.GCN1 = GCNConv(hidden_channels, hidden_channels)
        self.GCN2 = GCNConv(hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.GIN(x, edge_index)
        x = x.relu()
        x = self.GIN1(x, edge_index).relu()
        x = x.relu()
        x = self.GIN2(x, edge_index)

        # x = self.GCN(x, edge_index).relu()
        # x = self.GCN1(x, edge_index).relu()
        # x = self.GCN2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x


