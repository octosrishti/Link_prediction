
import torch
import shap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from models import GCN
from torch_geometric.data import Data


def explain_gnn(model, data):
    model.eval()

    # Use a non-GUI backend for Matplotlib to prevent threading issues
    matplotlib.use("Agg")

    # Function to generate model predictions for SHAP
    def model_forward(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to("cpu")  # Ensure on CPU
        with torch.no_grad():
            return model(x_tensor, data.edge_index.to("cpu")).detach().numpy()

    # Ensure max_evals is at least 2 * num_features + 1
    num_features = data.x.shape[1]
    max_evals = max(100, 2 * num_features + 1)  # Adjust max_evals

    # Create SHAP PermutationExplainer with updated max_evals
    explainer = shap.PermutationExplainer(
        model_forward, data.x.cpu().numpy(), max_evals=max_evals
    )
    shap_values = explainer(data.x.cpu().numpy())

    # Feature names for visualization
    feature_names = np.array([f"Feature {i}" for i in range(data.x.shape[1])])

    # Plot SHAP summary bar chart
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, data.x.cpu().numpy(), feature_names=feature_names, plot_type="bar"
    )
    plt.savefig("shap_summary.png")  # Save the plot to a file
    print("SHAP summary plot saved as shap_summary.png")
