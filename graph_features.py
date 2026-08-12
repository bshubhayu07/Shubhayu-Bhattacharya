"""
graph_features.py
-------------------
Builds a NetworkX transaction graph from raw account/transaction data, and
computes per-node features that combine:
  (a) behavioral signals (transaction counts, amounts, velocity)
  (b) structural signals (degree, cycle membership, betweenness) -- the
      features a plain tabular model (no graph awareness) could never see.

Also exposes build_pyg_data() which converts everything into a PyTorch
Geometric Data object ready for GNN training/inference.
"""

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

NODE_FEATURE_COLUMNS = [
    "account_age_days_norm",
    "kyc_verified",
    "in_degree",
    "out_degree",
    "total_in_amount",
    "total_out_amount",
    "pass_through_ratio",       # out_amount / in_amount -- close to 1.0 is suspicious (mule behavior)
    "avg_hop_latency_minutes",   # avg time between receiving and sending money -- low = fast pass-through
    "in_cycle",                    # is this node part of any directed cycle length <= 8?
    "clustering_coefficient",
]


def build_graph(accounts: pd.DataFrame, transactions: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in accounts.iterrows():
        G.add_node(row["account_id"], **row.to_dict())
    for _, row in transactions.iterrows():
        # DiGraph keeps only the latest parallel edge by default; aggregate multi-edges instead
        if G.has_edge(row["src"], row["dst"]):
            G[row["src"]][row["dst"]]["amount"] += row["amount"]
            G[row["src"]][row["dst"]]["count"] += 1
        else:
            G.add_edge(row["src"], row["dst"], amount=row["amount"], count=1,
                       timestamp=row["timestamp"])
    return G


def _detect_cycle_membership(G: nx.DiGraph, max_len=8) -> set:
    """Returns the set of node IDs that belong to at least one directed cycle
    of length <= max_len. Used as a structural fraud-ring signal (circular
    layering rings are, by construction, cycles)."""
    in_cycle = set()
    try:
        for scc in nx.strongly_connected_components(G):
            if len(scc) >= 2:
                if len(scc) <= max_len:
                    in_cycle.update(scc)
                else:
                    sub = G.subgraph(scc)
                    for cycle in nx.simple_cycles(sub, length_bound=max_len):
                        in_cycle.update(cycle)
                        if len(in_cycle) >= len(G):
                            break
    except Exception:
        pass
    return in_cycle


def compute_node_features(G: nx.DiGraph, transactions: pd.DataFrame) -> pd.DataFrame:
    txns = transactions.copy()
    txns["timestamp"] = pd.to_datetime(txns["timestamp"], format="mixed", utc=True)

    in_cycle_nodes = _detect_cycle_membership(G)
    clustering = nx.clustering(G.to_undirected())

    # Pre-group transactions by destination and source to optimize lookup performance
    in_edges_dict = dict(list(txns.groupby("dst")))
    out_edges_dict = dict(list(txns.groupby("src")))

    rows = []
    for node_id, attrs in G.nodes(data=True):
        in_edges = in_edges_dict.get(node_id, pd.DataFrame())
        out_edges = out_edges_dict.get(node_id, pd.DataFrame())

        total_in = in_edges["amount"].sum() if not in_edges.empty else 0.0
        total_out = out_edges["amount"].sum() if not out_edges.empty else 0.0
        pass_through_ratio = (total_out / total_in) if total_in > 0 else 0.0

        # average time between receiving money and the next outgoing transaction
        # (fast pass-through is a strong mule signal)
        hop_latencies = []
        if not in_edges.empty and not out_edges.empty:
            in_times = in_edges["timestamp"].sort_values().tolist()
            out_times = out_edges["timestamp"].sort_values().tolist()
            for t_in in in_times:
                later_outs = [t for t in out_times if t >= t_in]
                if later_outs:
                    hop_latencies.append((min(later_outs) - t_in).total_seconds() / 60.0)
        avg_hop_latency = float(np.mean(hop_latencies)) if hop_latencies else 999999.0  # large = no fast pass-through

        rows.append({
            "account_id": node_id,
            "account_age_days_norm": min(attrs.get("account_age_days", 0) / 1500.0, 1.0),
            "kyc_verified": float(attrs.get("kyc_verified", False)),
            "in_degree": G.in_degree(node_id),
            "out_degree": G.out_degree(node_id),
            "total_in_amount": total_in,
            "total_out_amount": total_out,
            "pass_through_ratio": min(pass_through_ratio, 3.0),  # clip extreme outliers
            "avg_hop_latency_minutes": min(avg_hop_latency, 999999.0),
            "in_cycle": float(node_id in in_cycle_nodes),
            "clustering_coefficient": clustering.get(node_id, 0.0),
        })

    return pd.DataFrame(rows)


def build_pyg_data(accounts: pd.DataFrame, transactions: pd.DataFrame, labels: pd.DataFrame = None):
    """
    Builds everything needed for GNN training/inference:
      - the NetworkX graph (for structural queries / visualization)
      - the node feature DataFrame (human-readable)
      - a PyTorch Geometric Data object (for the model)
      - a node_id <-> index mapping (PyG needs integer node indices)
    """
    G = build_graph(accounts, transactions)
    node_features_df = compute_node_features(G, transactions)

    # normalize the wide-range monetary/latency features (log1p + clip) so the
    # GNN isn't dominated by raw rupee amounts
    nf = node_features_df.copy()
    nf["total_in_amount"] = np.log1p(nf["total_in_amount"])
    nf["total_out_amount"] = np.log1p(nf["total_out_amount"])
    nf["avg_hop_latency_minutes"] = np.log1p(nf["avg_hop_latency_minutes"].clip(upper=100000))

    node_ids = nf["account_id"].tolist()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    x = torch.tensor(nf[NODE_FEATURE_COLUMNS].to_numpy(), dtype=torch.float)

    edges = [(id_to_idx[u], id_to_idx[v]) for u, v in G.edges() if u in id_to_idx and v in id_to_idx]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)

    y = None
    if labels is not None:
        label_map = labels.set_index("account_id")["is_fraud_ring"].to_dict()
        y = torch.tensor([int(label_map.get(nid, False)) for nid in node_ids], dtype=torch.long)
        data.y = y

    return {
        "graph": G,
        "node_features_df": node_features_df,
        "pyg_data": data,
        "node_ids": node_ids,
        "id_to_idx": id_to_idx,
    }

