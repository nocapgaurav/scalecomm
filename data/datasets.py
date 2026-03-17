"""
datasets.py
===========
Handles loading, preprocessing, and returning PyTorch Geometric Data objects
for multiple graph datasets used in community detection:
  - Cora
  - CiteSeer
  - Amazon Photo
  - Amazon Computers
  - DBLP (Coauthor)
  - Facebook (SNAP) — manual load
  - Enron Email        — manual load

All datasets are automatically downloaded on first use via PyG's built-in loaders.
"""

import os
import torch
import numpy as np
import networkx as nx
from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    Coauthor,
)
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_DATASETS = [
    "cora", "citeseer", "amazon-photo", "amazon-computers",
    "dblp", "facebook", "enron"
]

DATA_ROOT = os.path.join(os.path.dirname(__file__), "raw")


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_dataset(name: str, root: str = DATA_ROOT):
    """
    Load a graph dataset by name and return a PyG Data object.

    Parameters
    ----------
    name : str
        One of: 'cora', 'citeseer', 'amazon-photo', 'amazon-computers',
                'dblp', 'facebook', 'enron'
    root : str
        Directory where raw data is stored / will be downloaded.

    Returns
    -------
    data : torch_geometric.data.Data
        Graph with attributes:
          - data.x          : Node feature matrix [N, F]
          - data.edge_index : COO edge index [2, E]
          - data.y          : Ground-truth community labels [N]
          - data.num_classes: Number of communities
    """
    name = name.lower().strip()

    if name == "cora":
        return _load_planetoid("Cora", root)

    elif name == "citeseer":
        return _load_planetoid("CiteSeer", root)

    elif name == "amazon-photo":
        return _load_amazon("Photo", root)

    elif name == "amazon-computers":
        return _load_amazon("Computers", root)

    elif name == "dblp":
        return _load_coauthor("DBLP", root)

    elif name == "facebook":
        return _load_facebook(root)

    elif name == "enron":
        return _load_enron(root)

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Supported: {SUPPORTED_DATASETS}"
        )


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def _load_planetoid(name: str, root: str):
    """Load Cora or CiteSeer via PyG Planetoid."""
    dataset = Planetoid(
        root=os.path.join(root, name),
        name=name,
        transform=NormalizeFeatures()
    )
    data = dataset[0]
    data.num_classes = dataset.num_classes
    print(f"[Dataset] {name} loaded.")
    print(f"  Nodes      : {data.num_nodes}")
    print(f"  Edges      : {data.num_edges}")
    print(f"  Features   : {data.num_node_features}")
    print(f"  Communities: {data.num_classes}")
    return data


def _load_amazon(name: str, root: str):
    """Load Amazon Photo or Computers via PyG Amazon."""
    dataset = Amazon(
        root=os.path.join(root, f"Amazon-{name}"),
        name=name,
        transform=NormalizeFeatures()
    )
    data = dataset[0]
    data.num_classes = dataset.num_classes
    print(f"[Dataset] Amazon-{name} loaded.")
    print(f"  Nodes      : {data.num_nodes}")
    print(f"  Edges      : {data.num_edges}")
    print(f"  Features   : {data.num_node_features}")
    print(f"  Communities: {data.num_classes}")
    return data


def _load_coauthor(name: str, root: str):
    """Load DBLP Coauthor network via PyG Coauthor."""
    dataset = Coauthor(
        root=os.path.join(root, f"Coauthor-{name}"),
        name=name,
        transform=NormalizeFeatures()
    )
    data = dataset[0]
    data.num_classes = dataset.num_classes
    print(f"[Dataset] Coauthor-{name} loaded.")
    print(f"  Nodes      : {data.num_nodes}")
    print(f"  Edges      : {data.num_edges}")
    print(f"  Features   : {data.num_node_features}")
    print(f"  Communities: {data.num_classes}")
    return data


def _load_facebook(root: str):
    """
    Load Facebook SNAP dataset (ego networks).
    File: facebook_combined.txt (edge list)
    Download: https://snap.stanford.edu/data/ego-Facebook.html

    Expected file path: data/raw/facebook/facebook_combined.txt
    Node features are identity matrix (no attributes).
    Ground-truth labels from circle memberships are approximated via
    Louvain community detection since the SNAP ego format is complex.
    """
    fb_path = os.path.join(root, "facebook", "facebook_combined.txt")

    if not os.path.exists(fb_path):
        raise FileNotFoundError(
            f"Facebook dataset not found at {fb_path}.\n"
            "Download from: https://snap.stanford.edu/data/ego-Facebook.html\n"
            "Place 'facebook_combined.txt' in data/raw/facebook/"
        )

    return _load_edgelist_with_louvain(fb_path, name="Facebook")


def _load_enron(root: str):
    """
    Load Enron Email dataset.
    File: Email-Enron.txt (edge list)
    Download: https://snap.stanford.edu/data/email-Enron.html

    Expected file path: data/raw/enron/Email-Enron.txt
    """
    enron_path = os.path.join(root, "enron", "Email-Enron.txt")

    if not os.path.exists(enron_path):
        raise FileNotFoundError(
            f"Enron dataset not found at {enron_path}.\n"
            "Download from: https://snap.stanford.edu/data/email-Enron.html\n"
            "Place 'Email-Enron.txt' in data/raw/enron/"
        )

    return _load_edgelist_with_louvain(enron_path, name="Enron", comment="#")


def _load_edgelist_with_louvain(filepath: str, name: str, comment=None):
    """
    Load an edge-list file into a PyG Data object.
    Node features = degree-based (log-normalized degree vector).
    Labels = Louvain community assignments (via NetworkX).
    """
    import networkx as nx
    from torch_geometric.utils import from_networkx
    from community import best_partition  # python-louvain

    # Build NetworkX graph
    kwargs = {"comments": comment} if comment else {}
    G = nx.read_edgelist(filepath, nodetype=int, **kwargs)
    G = nx.convert_node_labels_to_integers(G)

    print(f"[Dataset] {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Louvain communities as pseudo ground-truth labels
    partition = best_partition(G)
    labels = [partition[n] for n in range(G.number_of_nodes())]
    num_classes = max(labels) + 1

    # Simple structural features: log-degree + clustering coefficient
    degrees = np.array([G.degree(n) for n in range(G.number_of_nodes())], dtype=np.float32)
    log_deg = np.log1p(degrees).reshape(-1, 1)
    clust   = np.array([nx.clustering(G, n) for n in range(G.number_of_nodes())],
                        dtype=np.float32).reshape(-1, 1)
    features = np.concatenate([log_deg, clust], axis=1)

    data = from_networkx(G)
    data.x = torch.tensor(features, dtype=torch.float)
    data.y = torch.tensor(labels, dtype=torch.long)
    data.num_classes = num_classes

    print(f"  Communities (Louvain): {num_classes}")
    return data


# ---------------------------------------------------------------------------
# Mini-batch subgraph sampler (GraphSAINT-style random walk)
# ---------------------------------------------------------------------------

def get_random_walk_loader(data, batch_size: int = 2000,
                            walk_length: int = 2, num_steps: int = 5):
    """
    Returns a PyG GraphSAINT RandomWalk data loader for scalable mini-batch
    training. Used instead of full-graph batching for large datasets.

    Parameters
    ----------
    data       : PyG Data object
    batch_size : Number of root nodes per walk
    walk_length: Number of hops per random walk
    num_steps  : Number of mini-batches per epoch

    Returns
    -------
    loader : GraphSAINTRandomWalkSampler
    """
    from torch_geometric.loader import GraphSAINTRandomWalkSampler

    loader = GraphSAINTRandomWalkSampler(
        data,
        batch_size=batch_size,
        walk_length=walk_length,
        num_steps=num_steps,
        sample_coverage=100,
        save_dir=None,
        num_workers=0,
    )
    return loader


# ---------------------------------------------------------------------------
# Utility: dataset info summary
# ---------------------------------------------------------------------------

def dataset_summary(data):
    """Print a summary of a loaded PyG Data object."""
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"  Nodes          : {data.num_nodes:,}")
    print(f"  Edges          : {data.num_edges:,}")
    print(f"  Node features  : {data.num_node_features}")
    print(f"  Communities    : {data.num_classes}")
    print(f"  Label coverage : {(data.y >= 0).sum().item():,} / {data.num_nodes:,}")
    print("=" * 50 + "\n")
