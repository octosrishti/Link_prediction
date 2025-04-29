
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.optim import Adam
from torch_geometric.transforms import RandomLinkSplit
import os
import networkx as nx
from torch_geometric.utils import from_networkx, negative_sampling


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=3):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
        self.attention_weights = None

    def forward(self, x, edge_index):
        x, (edge_index, alpha) = self.conv1(
            x, edge_index, return_attention_weights=True
        )
        self.attention_weights = alpha
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        source, target = edge_index
        return (z[source] * z[target]).sum(dim=-1)  # Dot product similarity

