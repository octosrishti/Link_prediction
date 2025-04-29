
import os

print(f"[INFO] Current working directory: {os.getcwd()}")
import torch
import asyncio
import optuna
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from models import GCN, GAT, Node2VecMLP
from utils import load_graph
from utils import load_model_once
from utils.evaluate import evaluate_link_prediction
from utils.explanation_utils import (
    common_neighbors,
    jaccard_similarity,
    preferential_attachment,
    embedding_similarity,
)
from models.train_all_models import train_model, train_node2vec

print(f"[DEBUG] train_model imported from: {train_model.__module__}")
# from utils import evaluate_link_prediction
from utils import visualize_attention
from utils import explain_gnn
from utils.optuna_hyperopt import objective


# Initialize FastAPI app
app = FastAPI()


# Allow WebSocket connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
training_results = {}
# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define request body models
class ModelTypeRequest(BaseModel):
    model_type: str


class PredictRequest(BaseModel):
    source: int
    target: int


# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    global G, train_data, val_data, test_data, model, data
    # Load graph
    G, train_data, val_data, test_data = load_graph("facebook_edges")
    data = train_data
    # Load model
    model = GCN(in_channels=64, hidden_channels=47, out_channels=64)
    model.load_state_dict(
        torch.load("models/facebook_gcn.pth", map_location=torch.device("cpu"))
    )
    model.eval()  # Set the model to evaluation mode


@app.websocket("/ws/train_metrics")
async def websocket_train_metrics(websocket: WebSocket):
    print("🔌 WebSocket connection request received")
    await manager.connect(websocket)

    try:
        # 👉 Receive the first message to get selected model
        data = await websocket.receive_json()
        if data.get("type") != "select_model" or "model" not in data:
            await websocket.send_json(
                {"error": "Missing 'type' as 'select_model' or 'model'"}
            )
            await websocket.close()
            return

        selected_model = data["model"]
        print(f"🎯 Selected model from client: {selected_model}")

        models = {
            "gcn": GCN(
                in_channels=train_data.x.shape[1], hidden_channels=64, out_channels=64
            ),
            "gat": GAT(
                in_channels=train_data.x.shape[1], hidden_channels=64, out_channels=64
            ),
            "node2vec_mlp": None
        }

        if selected_model not in models and selected_model != "node2vec_mlp":
            await websocket.send_json(
                {"error": f"Model '{selected_model}' not supported."}
            )
            await websocket.close()
            return

        if selected_model == "node2vec_mlp":
            model, collected_metrics = train_node2vec(
                train_data,
                embedding_dim=64,
                hidden_dim=128,
                epochs=50,
                lr=0.01,
                # websocket=websocket,
            )
        else:
            model = models[selected_model].to(device)
            # Train selected model
            model, collected_metrics = await train_model(
                model, train_data, websocket=websocket
            )

        # Broadcast metrics during training
        for epoch, metrics in enumerate(collected_metrics):
            await manager.broadcast(
                {
                    "model": selected_model,
                    "epoch": epoch,
                    "metrics": metrics,
                }
            )
            await asyncio.sleep(1)

        training_results[selected_model] = collected_metrics

    except Exception as e:
        print(f" WebSocket error: {e}")
        await websocket.close()


# Train model in the background
@app.post("/train")
async def train_model_fn(request: ModelTypeRequest, background_tasks: BackgroundTasks):
    _, train_data, val_data, test_data = load_graph()
    # Load model using load_model_once
    if request.model_type == "node2vec_mlp":
        # background_tasks.add_task(
        #     train_node2vec,
        #     train_data,
        #     embedding_dim=64,
        #     hidden_dim=128,
        #     epochs=50,
        #     lr=0.01,
        # )
        # training_results[request.model_type] = collected_metrics
        # return {
        #     "message": f"Training for {request.model_type.upper()} started in the background",
        #     "metrics": training_results[request.model_type],
        # }
        model, collected_metrics = train_node2vec(
            train_data,
            embedding_dim=64,
            hidden_dim=128,
            epochs=50,
            lr=0.01,
        )
        training_results[request.model_type] = collected_metrics
        return {
            "message": f"Training for {request.model_type.upper()} completed successfully",
            "metrics": training_results[request.model_type],
        }
    elif request.model_type in ["gcn", "gat"]:
        model_class = GCN if request.model_type == "gcn" else GAT
        model = model_class(in_channels=64, hidden_channels=47, out_channels=64)
        # background_tasksadd_task(train_model, model, train_data)
        # background_tasks.add_task(train_model, model, train_data, epochs=50, lr=0.01)
        model, collected_metrics = await train_model(
            model, train_data, epochs=50, lr=0.01
        )
        training_results[request.model_type] = collected_metrics

    else:
        raise HTTPException(status_code=400, detail="Invalid model type")
    return {
        "message": f"Training for {request.model_type.upper()} completed successfully",
        "metrics": training_results[request.model_type],
    }


# Predict link probability
@app.post("/predict")
def predict_link(request: PredictRequest):
    try:
        _, train_data, val_data, test_data = load_graph()
        model = load_model_once("gcn", train_data.x.shape[1]).to(device)
        model.eval()

        if (
            request.source >= train_data.x.shape[0]
            or request.target >= train_data.x.shape[0]
        ):
            raise HTTPException(
                status_code=400, detail="Source or target node index out of range"
            )

        with torch.no_grad():
            link_pred = model.predict_link(
                train_data.x[request.source].unsqueeze(0).to(device),
                train_data.x[request.target].unsqueeze(0).to(device),
            ).item()

        return {"Link Probability": link_pred}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# # Load graph data
# @app.get("/load_graph")
# def get_graph():
#     train_data, val_data, test_data = load_graph()
#     return {
#         "train_edges": train_data.edge_index.shape[1],
#         "val_edges": val_data.edge_index.shape[1],
#         "test_edges": test_data.edge_index.shape[1],
#     }


@app.get("/load_graph")
def get_graph():
    _, train_data, val_data, test_data = load_graph()

    # Get node data (you can customize this based on your dataset)
    nodes = [{"data": {"id": str(i)}} for i in range(train_data.num_nodes)]

    # Get edges data from train, val, and test splits
    edge_index = train_data.edge_index.numpy()
    edges = []
    for i in range(edge_index.shape[1]):
        source = str(edge_index[0][i])
        target = str(edge_index[1][i])
        edges.append({"data": {"source": source, "target": target}})

    return {
        "nodes": nodes,
        "edges": edges,
        "train_edges": train_data.edge_index.shape[1],
        "val_edges": val_data.edge_index.shape[1],
        "test_edges": test_data.edge_index.shape[1],
    }


# @app.post("/evaluate_link_prediction/")
# async def evaluate_endpoint(model_data: ModelTypeRequest):
#     try:
#         # Map model type to the correct dataset name
#         # if model_data.model_type == "gcn":
#         #     dataset_name = "facebook_edges"
#         # else:
#         #     dataset_name = model_data.model_type  # Default to the model type if it's not "gcn"
#         dataset_name = "facebook_edges"
#         # Load the graph and get data splits
#         G, train_data, val_data, test_data = load_graph(dataset_name)

#         # Extract in_channels from node features
#         in_channels = test_data.x.size(1)

#         # Load the model using your caching logic
#         model = load_model_once(model_data.model_type, in_channels=in_channels)

#         # Evaluate the model
#         metrics = evaluate_link_prediction(model, test_data)

#         return metrics
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
@app.post("/evaluate_link_prediction/")
async def evaluate_endpoint(model_data: ModelTypeRequest):
    try:
        print(f"Received model type: {model_data.model_type}")
        
        # Load the graph and splits
        dataset_name = "facebook_edges"
        print(f"Dataset name: {dataset_name}")
        G, train_data, val_data, test_data = load_graph(dataset_name)

        if test_data.x is None:
            raise ValueError("Node features (test_data.x) are missing or malformed.")
        
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        print(f"Test data size: {test_data.x.size()}")

        # Extract in_channels from node features
        in_channels = test_data.x.size(1)
        print(f"In channels: {in_channels}")

        # Load model
        if model_data.model_type == "node2vec_mlp":
            # Handle Node2VecMLP initialization separately
            model = Node2VecMLP(in_dim=64, hidden_dim=128)
            model.load_state_dict(torch.load("models/facebook_node2vec_mlp.pth", map_location=torch.device("cpu")))
            
            # Extract embeddings for source and target nodes
            src_emb = test_data.x[0].unsqueeze(0)  # Replace 0 with actual source node index
            target_emb = test_data.x[1].unsqueeze(0)  # Replace 1 with actual target node index
            
            # Ensure the model receives both arguments
            prediction_score = model(src_emb, target_emb).item()
            metrics = {"prediction_score": prediction_score}

        else:
            # For other models, use the existing logic
            model = load_model_once(model_data.model_type, in_channels=in_channels)
            metrics = evaluate_link_prediction(model, test_data)
        
        print(f"Metrics computed: {metrics}")
        return metrics

    except Exception as e:
        print(f"Error encountered: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Visualize GAT attention weights
@app.post("/visualize_attention")
def visualize_attention_endpoint(request: ModelTypeRequest):
    _, train_data, val_data, test_data = load_graph()
    # model = load_model_once(request.model_type, train_data.x.shape[1]).to(device)
    # hardcoded fix only for 'gat'
    if request.model_type == "gat":
        model = load_model_once("gat", in_channels=47).to(device)
    else:
        model = load_model_once(request.model_type, train_data.x.shape[1]).to(device)

    if request.model_type != "gat":
        raise HTTPException(
            status_code=400, detail="Visualization is only available for GAT model."
        )

    visualize_attention(model, train_data)
    # return {"message": "Attention weights visualized successfully"}
    return {
        "message": "Attention weights visualized successfully",
        "model": request.model_type,
        "num_nodes": train_data.num_nodes,
    }


# # Explain GNN predictions using SHAP
@app.post("/explain_gnn")
def explain_gnn_endpoint(request: ModelTypeRequest):
    _, train_data, val_data, test_data = load_graph()
    model = load_model_once(request.model_type, train_data.x.shape[1]).to(device)

    explain_gnn(model, train_data)
    return {"message": "GNN explanation generated successfully"}


# # Asynchronous hyperparameter tuning with Optuna
@app.post("/optuna_hyperopt")
async def optuna_hyperopt_endpoint():
    study = optuna.create_study(direction="minimize")
    await asyncio.to_thread(study.optimize, objective, n_trials=20)
    return {"best_hyperparameters": study.best_params}


# @app.get("/predict_explain")
# async def predict_with_explanation(src: int, dst: int):
#     # G, train_data, val_data, test_data = load_graph("facebook_edges")
#     # model = GCN(in_channels=64, hidden_channels=64, out_channels=32)
#     # model.load_state_dict(torch.load("models/gcn_weights.pth"))
#     model.eval()
#     with torch.no_grad():
#         # Step 1: Get node embeddings
#         x, edge_index = data.x, data.edge_index
#         z = model(x, edge_index)

#         # Step 2: Get prediction
#         src_emb = z[src].unsqueeze(0)
#         dst_emb = z[dst].unsqueeze(0)
#         prediction = model.predict_link(src_emb, dst_emb).item()

#         # Step 3: Explanations
#         explanation = {
#             "Common Neighbors": common_neighbors(G, src, dst),
#             "Jaccard Similarity": jaccard_similarity(G, src, dst),
#             "Preferential Attachment": preferential_attachment(G, src, dst),
#             "Embedding Similarity": embedding_similarity(z, src, dst),
#         }

#         return {
#             "source": src,
#             "target": dst,
#             "prediction_score": prediction,
#             "explanation": explanation,
#         }







# # Load model
@app.post("/load_model")
def load_model_endpoint(request: ModelTypeRequest):
    _, train_data, val_data, test_data = load_graph()
    model = load_model_once(request.model_type, train_data.x.shape[1]).to(device)
    return {"message": f"{request.model_type.upper()} model loaded successfully"}


from fastapi import Query

def predict_link(src_emb, dst_emb):
    return torch.sigmoid((src_emb * dst_emb).sum(dim=1))

@app.get("/predict_explain")
async def predict_with_explanation(
    src: int,
    dst: int,
    model_type: str = Query("gcn", alias="model_type")
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G, train_data, val_data, test_data = load_graph("facebook_edges")

    if model_type.lower() == "gcn":
        model = GCN(in_channels=64, hidden_channels=47, out_channels=64)
        model.load_state_dict(torch.load("models/facebook_gcn.pth", map_location=device))
    elif model_type.lower() == "gat":
        model = GAT(in_channels=64, hidden_channels=47, out_channels=64)
        model.load_state_dict(torch.load("models/facebook_gat.pth", map_location=device))
    elif model_type.lower() == "node2vec":
        model = Node2VecMLP(embedding_dim=64, hidden_dim=128)
        model.load_state_dict(torch.load("models/facebook_node2vec_mlp.pth", map_location=device))
    else:
        raise ValueError(f"Unknown model: {model_type}")

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        x, edge_index = train_data.x.to(device), train_data.edge_index.to(device)
        z = model(x, edge_index)

        src_emb = z[src].unsqueeze(0)
        dst_emb = z[dst].unsqueeze(0)

        # ⬇️ Use common predict_link function instead of model.predict_link
        prediction = predict_link(src_emb, dst_emb).item()

        explanation = {
            "Common Neighbors": common_neighbors(G, src, dst),
            "Jaccard Similarity": jaccard_similarity(G, src, dst),
            "Preferential Attachment": preferential_attachment(G, src, dst),
            "Embedding Similarity": embedding_similarity(z, src, dst),
        }

        return {
            "source": src,
            "target": dst,
            "model": model_type,
            "prediction_score": prediction,
            "explanation": explanation,
        }





# Run FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
