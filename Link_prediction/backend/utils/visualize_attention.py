
import sys
# import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import matplotlib.pyplot as plt
# from load_model_once import load_model_once
from utils.load_model_once import load_model_once
from torch_geometric.data import Data


def visualize_attention(model, data):
    """Generate and display a histogram of attention weights"""
    model.eval()
    with torch.no_grad():
        model(data.x, data.edge_index)  # Forward pass

    # Extract attention weights
    if model.attention_weights is None:
        print("[ERROR] No attention weights found in the model!")
        return

    att_weights = model.attention_weights.detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.hist(
        att_weights.flatten(), bins=50, alpha=0.75, edgecolor="black", color="blue"
    )
    plt.xlabel("Attention Weight")
    plt.ylabel("Frequency")
    plt.title("Histogram of GAT Attention Weights")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="gat", help="Choose the model type"
    )
    args = parser.parse_args()

    model_type = args.model
    model = load_model_once(model_type, in_channels=47)

    if "gat" not in model_type:
        print("[ERROR] You need to load a GAT model for attention visualization!")
        exit()

    num_nodes = 10
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]],
        dtype=torch.long,
    )
    x = torch.randn((num_nodes, 64))
    data = Data(x=x, edge_index=edge_index)

    visualize_attention(model, data)
