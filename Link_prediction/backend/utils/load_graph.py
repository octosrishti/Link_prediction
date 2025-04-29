import os
import networkx as nx
import torch
from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.transforms import RandomLinkSplit


def load_graph(dataset="facebook_edges"):
    """Loads and processes a graph dataset."""
    file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data", f"{dataset}.txt")
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found!")

    # Load Graph
    G = nx.read_edgelist(file_path, nodetype=int)
    print(
        f"Loaded {dataset} dataset with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )

    # Convert to PyG Data
    data = from_networkx(G)
    data.x = torch.randn((data.num_nodes, 64))  # Random node features
    data.edge_index = to_undirected(data.edge_index)

    # Train/Test Split
    transform = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True)
    train_data, val_data, test_data = transform(data)

    # Ensure edge_label_index is properly handled
    if hasattr(train_data, "edge_label") and hasattr(train_data, "edge_label_index"):
        pos_edge_label_index = train_data.edge_label_index[
            :, train_data.edge_label == 1
        ]
        neg_edge_label_index = train_data.edge_label_index[
            :, train_data.edge_label == 0
        ]
        train_data.edge_label_index = torch.cat(
            [pos_edge_label_index, neg_edge_label_index], dim=1
        )
        train_data.train_pos_edge_index = pos_edge_label_index  
    else:
        raise ValueError(
            "`train_data` is missing `edge_label` or `edge_label_index`. Check dataset processing!"
        )

    # Ensure the attribute is present before saving
    print(f"Attributes before saving: {list(train_data.keys())}")

    # Save custom attributes
    custom_attrs = {"train_pos_edge_index": train_data.train_pos_edge_index}

    # Print and verify split data details
    print(
        f"Train Edges: {train_data.edge_index.shape[1]}, Val Edges: {val_data.edge_index.shape[1]}, Test Edges: {test_data.edge_index.shape[1]}"
    )

    # Save Processed Data
    save_path = os.path.abspath(f"../data/{dataset}_processed.pt")
    torch.save((train_data, val_data, test_data, custom_attrs), save_path)
    print(f" Processed data saved to {save_path}")

    # return train_data, val_data, test_data
    return G, train_data, val_data, test_data



if __name__ == "__main__":
    dataset = "facebook_edges"
    os.makedirs(os.path.dirname("../data/facebook_edges_processed.pt"), exist_ok=True)

    train_data, val_data, test_data = load_graph(dataset)
