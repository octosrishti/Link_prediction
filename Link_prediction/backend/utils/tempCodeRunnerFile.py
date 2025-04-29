import argparse
import torch
from torch_geometric.data import Data
from utils.load_model_once import load_model_once

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="gcn",
    choices=["gcn", "gat", "node2vec_mlp"],
    help="Choose the model type",
)
args = parser.parse_args()

# Load the selected model
model_type = args.model
in_channels = (
    64 if model_type != "node2vec_mlp" else 32
)  # Node2Vec uses a different dimension
model = load_model_once(model_type, in_channels)

# Create dummy graph data
num_nodes = 10
edge_index = torch.tensor(
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]], dtype=torch.long
)
x = torch.randn((num_nodes, in_channels))  # Random node features
data = Data(x=x, edge_index=edge_index)

# Run inference
model.eval()
with torch.no_grad():
    output = model(data.x, data.edge_index)

print(f"[INFO] Model ({model_type}) output:", output)
