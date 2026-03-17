"""
graph_utils.py
==============
Utility functions for graph preprocessing and analysis in ScaleComm.

Includes:
  - add_self_loops_if_missing : Ensure self-loops exist in edge_index
  - compute_degree_features   : Add degree as extra node feature
  - compute_laplacian_pe      : Laplacian positional encoding for nodes
  - split_train_val_test      : Random train/val/test node splits
  - visualize_communities     : (optional) Plot community assignments
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_undirected,
    degree,
    to_networkx,
)


# ---------------------------------------------------------------------------
# Graph preprocessing
# ---------------------------------------------------------------------------

def preprocess_graph(data, add_self_loops_flag: bool = True,
                     to_undirected_flag: bool = True):
    """
    Apply standard preprocessing to a PyG Data object.

    Steps:
      1. Convert to undirected graph (symmetrize edges)
      2. Remove any existing self-loops
      3. Optionally add self-loops (helps GAT aggregation)

    Parameters
    ----------
    data               : PyG Data object
    add_self_loops_flag: Whether to add self-loops
    to_undirected_flag : Whether to force undirected edges

    Returns
    -------
    data : Preprocessed PyG Data object
    """
    edge_index = data.edge_index

    if to_undirected_flag:
        edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

    edge_index, _ = remove_self_loops(edge_index)

    if add_self_loops_flag:
        edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)

    data.edge_index = edge_index
    return data


def augment_features_with_degree(data):
    """
    Append log-degree as an additional node feature.

    Node degree captures structural position information that is
    complementary to attribute-based features.

    Parameters
    ----------
    data : PyG Data object with data.x and data.edge_index

    Returns
    -------
    data : Data object with augmented data.x (one extra column)
    """
    row = data.edge_index[0]
    deg = degree(row, num_nodes=data.num_nodes, dtype=torch.float)
    log_deg = torch.log1p(deg).unsqueeze(1)  # [N, 1]

    data.x = torch.cat([data.x, log_deg], dim=1)
    return data


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

def compute_laplacian_pe(data, k: int = 16):
    """
    Compute Laplacian Positional Encoding (LapPE) for nodes.

    Uses the bottom-k non-trivial eigenvectors of the normalized graph
    Laplacian as positional features. These capture global graph structure
    and improve community detection in graphs with weak attribute signals.

    Parameters
    ----------
    data : PyG Data object
    k    : Number of eigenvectors to use (default 16)

    Returns
    -------
    pe : Positional encoding tensor [N, k]
    """
    N = data.num_nodes
    k = min(k, N - 2)  # Can't compute more than N-2 eigenvectors

    # Build NetworkX graph for eigendecomposition
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Normalized Laplacian
    L = nx.normalized_laplacian_matrix(G).toarray().astype(np.float32)

    # Eigendecomposition (use scipy for sparse efficiency on large graphs)
    try:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        L_sparse = csr_matrix(L)
        eigenvalues, eigenvectors = eigsh(L_sparse, k=k + 1,
                                          which="SM", tol=1e-5)
        # Skip trivial eigenvector (eigenvalue ≈ 0)
        eigenvectors = eigenvectors[:, 1:k + 1]
    except Exception:
        # Fallback to dense eigensolver for small graphs
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        eigenvectors = eigenvectors[:, 1:k + 1]

    pe = torch.tensor(eigenvectors, dtype=torch.float)
    return pe


# ---------------------------------------------------------------------------
# Train / Val / Test splits
# ---------------------------------------------------------------------------

def random_node_split(num_nodes: int, train_ratio: float = 0.6,
                      val_ratio: float = 0.2, seed: int = 42):
    """
    Create random train/val/test index splits for nodes.

    Parameters
    ----------
    num_nodes  : Total number of nodes
    train_ratio: Fraction for training
    val_ratio  : Fraction for validation
    seed       : Random seed

    Returns
    -------
    train_idx, val_idx, test_idx : torch.Tensor (1D index tensors)
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(num_nodes)

    n_train = int(num_nodes * train_ratio)
    n_val   = int(num_nodes * val_ratio)

    train_idx = torch.tensor(idx[:n_train], dtype=torch.long)
    val_idx   = torch.tensor(idx[n_train:n_train + n_val], dtype=torch.long)
    test_idx  = torch.tensor(idx[n_train + n_val:], dtype=torch.long)

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------

def graph_stats(data) -> dict:
    """
    Compute basic graph statistics for a PyG Data object.

    Returns
    -------
    stats : dict with keys: num_nodes, num_edges, avg_degree,
                            density, num_isolated, is_connected
    """
    N = data.num_nodes
    E = data.num_edges

    row = data.edge_index[0]
    deg = degree(row, num_nodes=N, dtype=torch.float)

    stats = {
        "num_nodes"    : N,
        "num_edges"    : E,
        "avg_degree"   : float(deg.mean().item()),
        "max_degree"   : int(deg.max().item()),
        "density"      : E / (N * (N - 1)) if N > 1 else 0.0,
        "num_isolated" : int((deg == 0).sum().item()),
    }

    return stats


def print_graph_stats(data):
    """Print formatted graph statistics."""
    stats = graph_stats(data)
    print("\n" + "-" * 40)
    print("  GRAPH STATISTICS")
    print("-" * 40)
    for k, v in stats.items():
        print(f"  {k:20s}: {v}")
    print("-" * 40 + "\n")


# ---------------------------------------------------------------------------
# Community visualization (requires matplotlib)
# ---------------------------------------------------------------------------

def visualize_communities(data, labels, title: str = "Community Detection",
                           max_nodes: int = 500, save_path: str = None):
    """
    Visualize detected communities using NetworkX spring layout.

    Only practical for small graphs (< 500 nodes).

    Parameters
    ----------
    data      : PyG Data object
    labels    : Predicted community labels [N]
    title     : Plot title
    max_nodes : Subsample to this many nodes for clarity
    save_path : If given, save plot to this path
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("[Visualization] matplotlib not installed. Skipping.")
        return

    N = data.num_nodes
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Subsample for visualization
    if N > max_nodes:
        idx = np.random.choice(N, max_nodes, replace=False)
        print(f"[Visualization] Subsampling {max_nodes}/{N} nodes for clarity.")
    else:
        idx = np.arange(N)

    # Build subgraph
    G = to_networkx(data, to_undirected=True)
    sub_G = G.subgraph(idx.tolist())

    # Community colors
    sub_labels = labels[idx]
    K = int(sub_labels.max()) + 1
    colors = cm.tab20(np.linspace(0, 1, K))
    node_colors = [colors[sub_labels[i] % K] for i in range(len(idx))]

    # Draw
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(sub_G, seed=42, k=2.0 / np.sqrt(len(idx)))
    nx.draw_networkx(
        sub_G, pos=pos,
        node_color=node_colors,
        node_size=30,
        with_labels=False,
        edge_color="gray",
        alpha=0.7,
        width=0.3,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualization] Saved to {save_path}")
    else:
        plt.show()

    plt.close()
