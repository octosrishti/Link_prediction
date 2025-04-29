import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.link_predictor = nn.Sequential(
            nn.Linear(2 * out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.init_weights()

    def init_weights(self):
        for m in self.link_predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        source_features = z[edge_index[0]]
        target_features = z[edge_index[1]]
        combined_features = torch.cat([source_features, target_features], dim=1)
        link_pred = self.link_predictor(combined_features)
        return torch.sigmoid(link_pred)


    def predict_link(self, source_embedding, target_embedding):
        combined_features = torch.cat([source_embedding, target_embedding], dim=1)
        return torch.sigmoid(self.link_predictor(combined_features))
