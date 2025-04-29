
import os
import sys
import argparse
import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from evaluate import evaluate_link_prediction  # Import evaluation function
from load_graph import load_graph 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from load_model_once import load_model_once

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "node2vec_mlp"])
args = parser.parse_args()

# Load the selected model
model_type = args.model
in_channels = 64  # Set to 64 for Node2VecMLP
model = load_model_once(model_type, in_channels)
dataset = "facebook_edges"  # Specify your dataset name
_, _, val_data, test_data = load_graph(dataset)  # Load validation and test splits
data = test_data 

pos_edge_index = data.edge_index  # Use edge_index directly from data
print(f"[DEBUG] Positive Edge Index: {pos_edge_index}")
neg_edge_index = negative_sampling(
    edge_index= data.edge_index,
    num_nodes=data.num_nodes,
    num_neg_samples=data.edge_index.size(1)
)

data.edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
data.edge_label = torch.cat(
    [torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))]
)

# Debugging prints
print(f"[INFO] Model Type: {model_type}")
print(f"[INFO] Input Shape: {data.x.shape}")
print(f"[INFO] Edge Index Shape: {data.edge_index.shape}")

# Run inference
model.eval()
with torch.no_grad():
    if model_type == "node2vec_mlp":
        z = model(data.x)  # No edge_index needed
    else:
        z = model(data.x, data.edge_index)  # For GCN, GAT

print(f"[DEBUG] Model Output Shape: {z.shape}")
print(f"[DEBUG] Sample Output: {z[:5]}")  # First 5 node embeddings

# Evaluate Model Performance
metrics = evaluate_link_prediction(model, data)

# Print AUC & AP Scores
print(f"[INFO] Model ({model_type}) Evaluation - AUC: {metrics['AUC']:.4f}, AP: {metrics['AP']:.4f}")
