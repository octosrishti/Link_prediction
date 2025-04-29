from sklearn.metrics import roc_auc_score, average_precision_score
import torch

def evaluate_link_prediction(model, data):
    """Evaluate link prediction model."""
    model.eval()
    with torch.no_grad():
        # Generate node embeddings
        if "node2vec" in model.__class__.__name__.lower():
            z = model(data.x)  # Node2VecMLP doesn't use edge_index
        else:
            z = model(data.x, data.edge_index)  # GCN, GAT

        print(f"[DEBUG] Node embeddings shape: {z.shape}")

        # Compute link prediction scores
        pos_edge_index = data.edge_label_index[:, data.edge_label == 1]
        neg_edge_index = data.edge_label_index[:, data.edge_label == 0]
        pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
        neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

        print(f"[DEBUG] Positive scores (first 5): {pos_scores[:5]}")
        print(f"[DEBUG] Negative scores (first 5): {neg_scores[:5]}")

        # Compute AUC & AP
        y_true = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        y_pred = torch.cat([pos_scores, neg_scores])

        auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        ap = average_precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy())

        return {"AUC": auc, "AP": ap}  # Return both metrics
