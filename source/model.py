import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d
from torch.nn import Module, ModuleList 

from torch_geometric.nn import GINConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch_geometric.nn.pool.topk_pool import TopKPooling
from torch_geometric.nn.pool.asap import ASAPooling
from torch_geometric.nn.pool.pan_pool import PANPooling
from torch_geometric.nn.pool.sag_pool import SAGPooling

from torch_geometric.nn.glob.sort import global_sort_pool
from torch_geometric.nn.glob.set2set import Set2Set
from torch_geometric.nn.glob.gmt import GraphMultisetTransformer

def MLP(input_channels, hidden_channels):
    return nn.Sequential(Linear(input_channels, hidden_channels),
                         BatchNorm1d(hidden_channels),
                         ReLU(), 
                         Linear(hidden_channels, hidden_channels),
                         BatchNorm1d(hidden_channels))

class GCN_test(torch.nn.Module):
    def __init__(self, channels : int):
        super(GCN_test, self).__init__()
        input_channels, hidden_channels, output_channels = channels[0], channels[1], channels[-1]
        self.GIN = GINConv(MLP(input_channels, hidden_channels))
        self.GIN1 = GINConv(MLP(hidden_channels, hidden_channels))
        self.GIN2 = GINConv(MLP(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, output_channels)
        self.pool1 = SAGPooling(in_channels = hidden_channels)
        # self.pool2 = TopKPooling(in_channels = hidden_channels)

    def forward(self, x, edge_index, batch):
        x = self.GIN(x, edge_index).relu()
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch = batch)
        x = self.GIN1(x, edge_index).relu()
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch = batch)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        return x

class GCN(Module):
    def __init__(self, channels : list):
        super(GCN, self).__init__()
        self.layers = ModuleList([GINConv(MLP(channels[i], channels[i+1])) for i in range(len(channels)-2)])
        self.lin = Linear(sum(channels[1:-1]), channels[-1])
        
    def forward(self, x, edge_index, batch):
        n = torch.tensor([], device = x.device)
        for layer in self.layers:
            x = layer(x, edge_index)
            x = x.relu()
            m = global_mean_pool(x, batch)
            n = torch.cat((n, m), dim=1)


        # x = global_mean_pool(x, batch)
        n = F.dropout(n, p=0.2, training=self.training)
        n = self.lin(n)
        return n

# FIXME
class GCN_set2set(Module):
    def __init__(self, channels : list):
        super(GCN_set2set, self).__init__()
        self.layers = ModuleList([GINConv(MLP(channels[i], channels[i+1])) for i in range(len(channels)-2)])
        self.readout = Set2Set(channels[-2], processing_steps=4, num_layers=4, layer_norm=True)
        self.lin = Linear(channels[-2]*2, channels[-1])
    
    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = x.relu()

        x = self.readout(x, batch)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        return x

class GCN_GMT(Module):
    def __init__(self, channels : list):
        super(GCN_GMT, self).__init__()
        self.layers = ModuleList([GINConv(MLP(channels[i], channels[i+1])) for i in range(len(channels)-2)])
        self.readout = GraphMultisetTransformer(channels[-2], channels[-2], channels[-1])

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = x.relu()
        x = self.readout(x, batch, edge_index)
        return x
    
