import torch

def edge_importance_from_grad(model, data):
    model.eval()
    data.x.requires_grad_(True)
    out = model(data)               # edge logits
    loss = out[:,1].mean()          # encourage fraudulent edge logit
    loss.backward()
    # gradient norm on node embeddings as proxy -> map to edges by sum of norm on incident nodes
    src, dst = data.edge_index
    grad_norm = data.x.grad.norm(dim=1)
    edge_score = (grad_norm[src] + grad_norm[dst]).detach().cpu().numpy().tolist()
    return edge_score
