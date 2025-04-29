import networkx as nx
import torch.nn.functional as F

# These work with NetworkX graph `G`
def common_neighbors(G: nx.Graph, u: int, v: int):
    return len(list(nx.common_neighbors(G, u, v)))

def jaccard_similarity(G: nx.Graph, u: int, v: int):
    preds = nx.jaccard_coefficient(G, [(u, v)])
    for _, _, score in preds:
        return score
    return 0.0

def preferential_attachment(G: nx.Graph, u: int, v: int):
    preds = nx.preferential_attachment(G, [(u, v)])
    for _, _, score in preds:
        return score
    return 0.0

def embedding_similarity(embeddings, u, v):
    emb_u = embeddings[u]
    emb_v = embeddings[v]
    return F.cosine_similarity(emb_u.unsqueeze(0), emb_v.unsqueeze(0)).item()
