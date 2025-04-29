
import os
import torch
import optuna
from torch_geometric.data import Data
# from load_graph import load_graph
from utils.load_graph import load_graph
import sys
sys.path.append("c:/Users/lenovo/MProject/backend")
from models.gcn import GCN


# Allow PyG Data and DataEdgeAttr classes for unpickling
from torch_geometric.data.data import Data, DataEdgeAttr

torch.serialization.add_safe_globals([DataEdgeAttr])


def dict_to_data(d):
    """Convert dictionary to PyG Data object."""
    data = Data()
    for key, value in d.items():
        setattr(data, key, value)
    return data


def objective(trial):
    """Objective function for Optuna hyperparameter tuning."""
    hidden_dim = trial.suggest_int("hidden_dim", 8, 64)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    # Dataset path (Fixed to your local directory)
    data_path = os.path.abspath(
        "C:/Users/lenovo/MProject/data/facebook_edges_processed.pt"
    )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    # Load dataset
    loaded_data = torch.load(data_path, weights_only=False)
    train_data = dict_to_data(loaded_data[0])

    print(f"Attributes in train_data: {list(train_data.keys())}")

    if not hasattr(train_data, "num_features"):
        train_data.num_features = (
            train_data.x.shape[1] if hasattr(train_data, "x") else 0
        )

    if not hasattr(train_data, "train_pos_edge_index"):
        raise AttributeError("Dataset does not contain 'train_pos_edge_index'.")

    # Ensure labels exist
    edge_label_index = train_data.edge_label_index
    edge_label = train_data.edge_label

    if edge_label_index is None or edge_label is None:
        raise AttributeError(
            "Dataset must contain 'edge_label_index' and 'edge_label'."
        )

    # Initialize GCN model
    model = GCN(
        in_channels=train_data.num_features, hidden_channels=hidden_dim, out_channels=64
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()

        # Forward pass
        node_embeddings = model(train_data.x, train_data.edge_index)

        # Compute scores for edges
        edge_src = node_embeddings[edge_label_index[0]]
        edge_dst = node_embeddings[edge_label_index[1]]
        edge_scores = torch.sigmoid(
            (edge_src * edge_dst).sum(dim=1)
        )  # Dot product for similarity

        loss = criterion(
            edge_scores.view(-1), edge_label.float()
        )  # Ensure correct shape
        loss.backward()
        optimizer.step()

    return loss.item()


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("\n🎯 Best trial:")
    best_trial = study.best_trial
    print(f"🔹 Best Loss: {best_trial.value}")
    print(f"🔹 Best Hyperparameters: {best_trial.params}")

    import optuna.visualization as vis
    import plotly.io as pio

    # Increase plot size
    fig1 = vis.plot_optimization_history(study)
    fig1.update_layout(width=1000, height=600)  # Set plot size
    fig1.show()

    # Save as image (optional)
    fig1.write_image("optuna_optimization_history.png")

    # Hyperparameter importance visualization
    fig2 = vis.plot_param_importances(study)
    fig2.update_layout(width=1000, height=600)
    fig2.show()

    # Save as image (optional)
    fig2.write_image("optuna_param_importance.png")

    # Optional: Use dark mode for better visibility
    fig1.update_layout(template="plotly_dark")
    fig2.update_layout(template="plotly_dark")
