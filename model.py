import torch
from scgat import SCGATConv
from torch_geometric.nn import GATConv


class SimpleGAT(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int, heads:int, hidden_size:int):
        super(SimpleGAT, self).__init__()
        
        self.conv1 = SCGATConv(in_channels, hidden_size, heads)
        self.lin1 = torch.nn.Linear(hidden_size*heads, out_channels)
        self.act1 = torch.nn.ReLU()
    def forward(self, x):
        out = self.conv1(x.x, x.edge_index)
        out = self.lin1(out)
        return self.act1(out)




class BenchmarkGAT(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int, heads:int, hidden_size:int):
        super(BenchmarkGAT, self).__init__()
        
        self.conv1 = GATConv(in_channels, hidden_size, heads)
        self.lin1 = torch.nn.Linear(hidden_size*heads, out_channels)
        self.act1 = torch.nn.ReLU()
    def forward(self, x):
        out = self.conv1(x.x, x.edge_index)
        out = self.lin1(out)
        return self.act1(out)

