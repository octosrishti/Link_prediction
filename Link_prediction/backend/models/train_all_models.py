import torch
import os
import networkx as nx
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.utils import from_networkx, negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from node2vec import Node2Vec
from .gcn import GCN
from .gat import GAT
import numpy as np
from .node2vec_mlp import Node2VecMLP
from starlette.websockets import WebSocketState


async def train_model(model, train_data, epochs=50, lr=0.01, websocket=None):

    """Train GCN or GAT model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    collected_metrics = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model(train_data.x.to(device), train_data.edge_index.to(device))

        # Generate negative edges
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_index.size(1),
        ).to(device)

        # Decode for positive and negative edges
        pos_logits = model.decode(z, train_data.edge_index.to(device))
        neg_logits = model.decode(z, neg_edge_index)

        pos_labels = torch.ones(pos_logits.size(0), device=device)
        neg_labels = torch.zeros(neg_logits.size(0), device=device)
        labels = torch.cat([pos_labels, neg_labels])

        logits = torch.cat([pos_logits, neg_logits])

        #  Fix: Match shape with labels
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(
                f"{model.__class__.__name__} - Epoch {epoch}: Loss = {loss.item():.4f}"
            )
        metrics = {
            "epoch": epoch,
            "loss": round(loss.item(), 4),
            # "accuracy": compute_accuracy(train_data, model),  # Example function
            # "precision": compute_precision(train_data, model),
            # "recall": compute_recall(train_data, model),
        }

        collected_metrics.append(metrics)

        # Send metrics over WebSocket, if connected
        if websocket and websocket.client_state != WebSocketState.CONNECTED:
            print(f"WebSocket disconnected; stopping metrics streaming at epoch {epoch}.")
            break
        if websocket:
            try:
                await websocket.send_json(metrics)
            except Exception as e:
                print(f" WebSocket send failed: {e}")

    # Return trained model and all collected metrics
    return model, collected_metrics


def train_node2vec(train_data, embedding_dim=64, hidden_dim=128, epochs=50, lr=0.01):
    """Train Node2Vec+MLP model for link prediction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Node2VecMLP(embedding_dim, hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    model.train()
    collected_metrics = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Generate negative edges
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_index.size(1),
        ).to(device)

        # Get node embeddings
        node_embeddings = train_data.x.to(device)

        pos_source_emb = node_embeddings[train_data.edge_index[0]]
        pos_target_emb = node_embeddings[train_data.edge_index[1]]

        neg_source_emb = node_embeddings[neg_edge_index[0]]
        neg_target_emb = node_embeddings[neg_edge_index[1]]

        pos_logits = model(pos_source_emb, pos_target_emb).squeeze()
        neg_logits = model(neg_source_emb, neg_target_emb).squeeze()

        pos_labels = torch.ones(pos_logits.size(0), device=device)
        neg_labels = torch.zeros(neg_logits.size(0), device=device)
        labels = torch.cat([pos_labels, neg_labels])
        logits = torch.cat([pos_logits, neg_logits])

        
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Node2VecMLP - Epoch {epoch}: Loss = {loss.item():.4f}")
        
        metrices = {
            "epoch": epoch,
            "loss": round(loss.item(), 4),
        }   
        collected_metrics.append(metrices)

    return model, collected_metrics


def load_data():
    """Load graph data and preprocess."""
    dataset = "facebook_edges"
    data_path = os.path.join(os.path.dirname(__file__), "../data", f"{dataset}.txt")

    print(f"Checking dataset file at: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file {data_path} not found!")



    # Load graph from edge list
    G = nx.read_edgelist(data_path, nodetype=int)


    
    node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([model.wv[str(node)] for node in sorted(G.nodes())])



    data = from_networkx(G)

    data.x = torch.tensor(embeddings, dtype=torch.float32)  

    # Split into train, validation, and test sets
    transform = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True)
    train_data, val_data, test_data = transform(data)

    # return train_data
    return train_data, val_data, test_data


def main():
    """Main function to train all models."""
    # train_data = load_data()
    
    train_data, val_data, test_data = load_data()

    models = {
        "GCN": GCN(in_channels=64, hidden_channels=47, out_channels=64),
        "GAT": GAT(in_channels=64, hidden_channels=47, out_channels=64),
    }

    save_dir = os.path.join(os.path.dirname(__file__), "../models")
    os.makedirs(save_dir, exist_ok=True)

    # Train and save GCN and GAT
    for name, model in models.items():
        trained_model = train_model(model, train_data)
        save_path = os.path.join(save_dir, f"facebook_{name.lower()}.pth")
        torch.save(trained_model.state_dict(), save_path)
        print(f" {name} model saved to {save_path}")

    # Train and save Node2Vec+MLP
    node2vec_model = train_node2vec(train_data)
    node2vec_path = os.path.join(save_dir, "facebook_node2vec_mlp.pth")
    torch.save(node2vec_model.state_dict(), node2vec_path)
    print(f"Node2VecMLP model saved to {node2vec_path}")


if __name__ == "__main__":
    main()
