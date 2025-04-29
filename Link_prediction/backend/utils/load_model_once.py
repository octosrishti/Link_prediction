
import os
import torch
from fastapi import HTTPException
from models.gat import GAT
from models.gcn import GCN
from models.node2vec_mlp import Node2VecMLP


# Cache loaded models
loaded_models = {}

# Check if CUDA (GPU) is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


def load_model_once(model_type: str, in_channels: int):
    """
    Loads and caches the model to avoid reloading it multiple times.
    If already loaded, returns the cached model.
    """
    # If model is already loaded, return it
    if model_type in loaded_models:
        print(f"[INFO] Using cached model: {model_type}")
        return loaded_models[model_type]

    # Define model file path
    # model_path = os.path.join("models", f"facebook_{model_type}.pth")
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", f"facebook_{model_type}.pth")


    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model file '{model_path}' not found!")

    print(f"[INFO] Loading {model_type} model from {model_path}")

    # Initialize the correct model architecture
    if model_type == "gcn":
        model = GCN(in_channels=in_channels, hidden_channels=47, out_channels=64)
        print(f"Loaded model: {model_type}")
    elif model_type == "gat":
        model = GAT(in_channels=in_channels, hidden_channels=47, out_channels=64)
        print(f"Loaded model: {model_type}")
    elif model_type == "node2vec_mlp":
        model = Node2VecMLP(embedding_dim=64, hidden_dim=128)  # Fix dimensions
        print(f"Loaded model: {model_type}")
    else:
        raise ValueError(f"[ERROR] Invalid model type: {model_type}")

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Move model to the selected device (CPU/GPU)
    model.to(device)
    model.eval()

    print(f"[SUCCESS] {model_type} model loaded successfully!")

    # Cache the model for future use
    loaded_models[model_type] = model
    return model


# Debugging: Run the script directly to test loading
if __name__ == "__main__":
    try:
        model = load_model_once("gcn", in_channels=64)
        print("[INFO] Model loaded successfully and ready for inference!")
    except Exception as e:
        print(f"[ERROR] {e}")
