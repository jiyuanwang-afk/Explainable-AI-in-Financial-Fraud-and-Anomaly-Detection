import pandas as pd, networkx as nx, torch
from torch_geometric.data import Data

MERCHANT_MAP = {"food":0, "electronics":1, "apparel":2, "crypto":3, "travel":4, "wire":5}

def build_transaction_graph(csv_path: str):
    df = pd.read_csv(csv_path)
    df['merchant_code'] = df['merchant_type'].map(MERCHANT_MAP).fillna(0).astype(int)
    # Create undirected multi-graph collapsed to simple graph
    G = nx.from_pandas_edgelist(df, 'src', 'dst', edge_attr=['amount','time_gap','merchant_code','fraud'], create_using=nx.Graph())
    # node features: degree, weighted degree
    degrees = {n: G.degree(n) for n in G.nodes()}
    deg_vec = torch.tensor([[degrees[n]] for n in G.nodes()], dtype=torch.float)
    node_index = {n:i for i,n in enumerate(G.nodes())}
    # edges
    edges = [(node_index[u], node_index[v]) for u,v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # edge features & labels (edge-level fraud)
    edge_feats = []
    edge_labels = []
    for u,v in G.edges():
        attrs = G[u][v]
        edge_feats.append([attrs['amount'], attrs['time_gap'], float(attrs['merchant_code'])])
        edge_labels.append(int(attrs.get('fraud',0)))
    edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    y = torch.tensor(edge_labels, dtype=torch.long)
    # simple node feature repeated to match edge supervision (for GCN simplicity we keep node x)
    x = deg_vec
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
