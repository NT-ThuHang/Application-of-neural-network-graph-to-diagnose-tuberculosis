import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d
from torch.nn import Module, ModuleList, Sequential

from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool

def MLP(input_channels, hidden_channels):
    return Sequential(Linear(input_channels, hidden_channels),
                    ReLU(),
                    BatchNorm1d(hidden_channels),
                    Linear(hidden_channels, hidden_channels),
                    ReLU(),
                    BatchNorm1d(hidden_channels))

class GNN(Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(GNN, self).__init__()
        self.gins = ModuleList([GINConv(MLP(input_channels, hidden_channels[0]))])
        
        for i in range(len(hidden_channels)-1):
            self.gins.append(GINConv(MLP(hidden_channels[i], hidden_channels[i+1])))
        
        self.lin = Linear(input_channels+sum(hidden_channels), output_channels)
        self.bn = BatchNorm1d(input_channels+sum(hidden_channels))
        
    def forward(self, x : torch.Tensor, edge_index: torch.Tensor, batch):
        xs = [global_mean_pool(x, batch)]
        for gin in self.gins:
            x = gin(x, edge_index)
            xs.append(global_mean_pool(x, batch))
            
        x = torch.cat(xs, dim=-1)
        x = self.bn(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)
        return torch.squeeze(x, dim=-1)