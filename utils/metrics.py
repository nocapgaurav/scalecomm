"""
metrics.py
==========
Evaluation metrics for community detection quality.

Implements:
  - NMI   : Normalized Mutual Information
  - ARI   : Adjusted Rand Index
  - ACC   : Clustering Accuracy (via Hungarian algorithm)
  - F1    : Macro-averaged F1 Score
  - Modularity : Modularity Q score via NetworkX
  - Conductance: Average community conductance

All metric functions accept numpy arrays or torch tensors.
"""

import numpy as np
import torch
import warnings
import networkx as nx

from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    f1_score,
)
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(arr):
    """Convert torch.Tensor or list to numpy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.array(arr)


# ---------------------------------------------------------------------------
# NMI — Normalized Mutual Information
# ---------------------------------------------------------------------------

def compute_nmi(true_labels, pred_labels) -> float:
    """
    Normalized Mutual Information between true and predicted communities.

    Measures the shared information between two clustering assignments.
    Ranges [0, 1]; 1 = perfect match, 0 = no mutual information.

    Parameters
    ----------
    true_labels : Ground-truth community labels [N]
    pred_labels : Predicted community labels [N]

    Returns
    -------
    nmi : float in [0, 1]
    """
    true_labels = _to_numpy(true_labels)
    pred_labels = _to_numpy(pred_labels)

    return normalized_mutual_info_score(
        true_labels, pred_labels, average_method="arithmetic"
    )


# ---------------------------------------------------------------------------
# ARI — Adjusted Rand Index
# ---------------------------------------------------------------------------

def compute_ari(true_labels, pred_labels) -> float:
    """
    Adjusted Rand Index between true and predicted communities.

    Measures the similarity of two clusterings, corrected for chance.
    Ranges [-1, 1]; 1 = perfect match, 0 = random, negative = worse than random.

    Parameters
    ----------
    true_labels : Ground-truth community labels [N]
    pred_labels : Predicted community labels [N]

    Returns
    -------
    ari : float in [-1, 1]
    """
    true_labels = _to_numpy(true_labels)
    pred_labels = _to_numpy(pred_labels)

    return adjusted_rand_score(true_labels, pred_labels)


# ---------------------------------------------------------------------------
# ACC — Clustering Accuracy (Hungarian algorithm)
# ---------------------------------------------------------------------------

def compute_acc(true_labels, pred_labels) -> float:
    """
    Clustering accuracy using the Hungarian algorithm for optimal label matching.

    Since clustering labels are permutation-invariant, we find the best
    one-to-one mapping between predicted and true cluster IDs via the
    linear assignment problem.

    Parameters
    ----------
    true_labels : Ground-truth labels [N]
    pred_labels : Predicted labels [N]

    Returns
    -------
    acc : float in [0, 1]
    """
    true_labels = _to_numpy(true_labels).astype(int)
    pred_labels = _to_numpy(pred_labels).astype(int)

    N = len(true_labels)
    K_true = int(true_labels.max()) + 1
    K_pred = int(pred_labels.max()) + 1
    K = max(K_true, K_pred)

    # Build confusion matrix [K, K]
    confusion = np.zeros((K, K), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        confusion[t, p] += 1

    # Solve assignment problem: maximize matches
    row_ind, col_ind = linear_sum_assignment(-confusion)
    correct = confusion[row_ind, col_ind].sum()

    return correct / N


# ---------------------------------------------------------------------------
# F1 Score
# ---------------------------------------------------------------------------

def compute_f1(true_labels, pred_labels, average: str = "macro") -> float:
    """
    Macro-averaged F1 score for community membership.

    Measures the harmonic mean of precision and recall for each
    community, averaged across all communities.

    Parameters
    ----------
    true_labels : Ground-truth labels [N]
    pred_labels : Predicted labels [N]
    average     : 'macro', 'weighted', or 'micro'

    Returns
    -------
    f1 : float in [0, 1]
    """
    true_labels = _to_numpy(true_labels)
    pred_labels = _to_numpy(pred_labels)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f1_score(true_labels, pred_labels,
                        average=average, zero_division=0)


# ---------------------------------------------------------------------------
# Modularity
# ---------------------------------------------------------------------------

def compute_modularity(edge_index, pred_labels, num_nodes: int = None) -> float:
    """
    Compute modularity Q of the predicted community partition.

    Modularity measures the fraction of edges within communities minus
    the expected fraction in a random graph with the same degree sequence.

    Q ∈ [-1, 1]; values > 0.3 indicate meaningful community structure.

    Parameters
    ----------
    edge_index  : Edge index [2, E] (torch or numpy)
    pred_labels : Predicted community labels [N]
    num_nodes   : Total number of nodes (inferred if None)

    Returns
    -------
    Q : float — Modularity score
    """
    pred_labels = _to_numpy(pred_labels)

    if isinstance(edge_index, torch.Tensor):
        edges = edge_index.t().detach().cpu().numpy()
    else:
        edges = edge_index.T

    # Build NetworkX graph
    G = nx.Graph()
    if num_nodes is not None:
        G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    # Build community partition as list of sets
    num_communities = int(pred_labels.max()) + 1
    communities = [
        set(np.where(pred_labels == k)[0].tolist())
        for k in range(num_communities)
        if (pred_labels == k).sum() > 0
    ]

    if not communities:
        return 0.0

    try:
        Q = nx.algorithms.community.quality.modularity(G, communities)
    except Exception:
        Q = 0.0

    return float(Q)


# ---------------------------------------------------------------------------
# Conductance
# ---------------------------------------------------------------------------

def compute_conductance(edge_index, pred_labels, num_nodes: int = None) -> float:
    """
    Compute average conductance of detected communities.

    Conductance of a community C = (edges crossing boundary) /
                                    min(vol(C), vol(V\C))

    Lower conductance = better isolated communities.
    Range [0, 1]; lower is better.

    Parameters
    ----------
    edge_index  : Edge index [2, E]
    pred_labels : Predicted community labels [N]
    num_nodes   : Total number of nodes

    Returns
    -------
    avg_conductance : float — Average conductance across communities
    """
    pred_labels = _to_numpy(pred_labels)

    if isinstance(edge_index, torch.Tensor):
        edges = edge_index.t().detach().cpu().numpy()
    else:
        edges = edge_index.T

    G = nx.Graph()
    if num_nodes is not None:
        G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    num_communities = int(pred_labels.max()) + 1
    conductances = []

    for k in range(num_communities):
        community = set(np.where(pred_labels == k)[0].tolist())
        if len(community) == 0 or len(community) == G.number_of_nodes():
            continue
        try:
            cond = nx.conductance(G, community)
            if not np.isnan(cond):
                conductances.append(cond)
        except Exception:
            continue

    return float(np.mean(conductances)) if conductances else 0.0


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    true_labels,
    pred_labels,
    edge_index=None,
    num_nodes: int = None,
    verbose: bool = True,
) -> dict:
    """
    Compute all evaluation metrics in one call.

    Parameters
    ----------
    true_labels : Ground-truth community labels [N]
    pred_labels : Predicted community labels [N]
    edge_index  : Graph edges for modularity/conductance (optional)
    num_nodes   : Number of nodes (optional)
    verbose     : If True, print a formatted results table

    Returns
    -------
    results : dict with keys nmi, ari, acc, f1, modularity, conductance
    """
    results = {}

    results["nmi"]         = compute_nmi(true_labels, pred_labels)
    results["ari"]         = compute_ari(true_labels, pred_labels)
    results["acc"]         = compute_acc(true_labels, pred_labels)
    results["f1"]          = compute_f1(true_labels, pred_labels)

    if edge_index is not None:
        results["modularity"]  = compute_modularity(edge_index, pred_labels, num_nodes)
        results["conductance"] = compute_conductance(edge_index, pred_labels, num_nodes)
    else:
        results["modularity"]  = None
        results["conductance"] = None

    if verbose:
        _print_results(results)

    return results


def _print_results(results: dict):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 45)
    print("  COMMUNITY DETECTION — EVALUATION RESULTS")
    print("=" * 45)
    for metric, value in results.items():
        if value is None:
            print(f"  {metric.upper():15s}  : N/A")
        else:
            print(f"  {metric.upper():15s}  : {value:.4f}")
    print("=" * 45 + "\n")
