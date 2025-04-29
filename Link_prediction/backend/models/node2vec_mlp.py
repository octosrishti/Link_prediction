import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn


class Node2VecMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Node2VecMLP, self).__init__()
        self.fc1 = torch.nn.Linear(
            in_dim * 2, hidden_dim
        )  # Multiply by 2 for concatenation
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, source_emb, target_emb): 
        x = torch.cat([source_emb, target_emb], dim=1)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
