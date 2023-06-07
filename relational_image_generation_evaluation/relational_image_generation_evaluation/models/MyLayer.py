from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer, GATv2Conv
import torch

from models.MyGATv2Conv import MyGATv2Conv, MyGATv2Conv2

class EdgeModel(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim):
        super().__init__()
        self.lin_i = torch.nn.Linear(node_in_dim, edge_in_dim)
        self.lin_j = torch.nn.Linear(node_in_dim, edge_in_dim)
        self.lin_e = torch.nn.Linear(edge_in_dim, edge_in_dim)

    def forward(self, src, dest, edge_attr, u, batch):
        out = self.lin_e(edge_attr) + self.lin_i(src) + self.lin_j(dest)
        return out

class EdgeModel2(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim):
        super().__init__()
        self.lin_i = torch.nn.Linear(node_in_dim, edge_in_dim)
        self.lin_j = torch.nn.Linear(node_in_dim, edge_in_dim)
        self.lin_e = torch.nn.Linear(edge_in_dim, edge_in_dim)
        self.edge_in_dim = edge_in_dim

    def forward(self, src, dest, edge_attr, u, batch):
        edge_attr2 = edge_attr[:, self.edge_in_dim:]
        out = self.lin_e(edge_attr[:, :self.edge_in_dim]) + self.lin_i(src) + self.lin_j(dest)
        return torch.cat([out, edge_attr2], dim=1)

def construct_my_layer(node_in_dim, node_out_dim, edge_in_dim):
    layer = MetaLayer(
            EdgeModel(node_in_dim, edge_in_dim),
            MyGATv2Conv(in_channels=node_in_dim, out_channels=node_out_dim, edge_in_dim=edge_in_dim, heads=2, concat=False, fill_value=0)
        )
    return layer

def construct_my_layer2(node_in_dim, node_out_dim, edge_in_dim):
    layer = MetaLayer(
            EdgeModel2(node_in_dim, edge_in_dim),
            MyGATv2Conv2(in_channels=node_in_dim, out_channels=node_out_dim, edge_in_dim=edge_in_dim, heads=2, concat=False, fill_value=0)
        )
    return layer