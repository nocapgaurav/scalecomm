"""
evaluate.py
===========
Standalone evaluation script for ScaleComm community detection results.

Can be run in two modes:
  1. Evaluate from saved checkpoint + dataset:
     python training/evaluate.py --dataset cora --checkpoint checkpoints/scalecomm_cora_best.pt

  2. Evaluate from saved prediction labels only:
     python training/evaluate.py --dataset cora --labels outputs/pred_labels_cora.npy

Outputs all community detection metrics:
  - NMI, ARI, ACC, F1, Modularity, Conductance

Also generates:
  - Per-community size statistics
  - Community quality breakdown table
  - (Optional) Visualization of detected communities
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

from data.datasets import load_dataset
from models.gat_encoder import GATEncoder
from models.clustering import get_clusterer
from utils.metrics import evaluate_all, compute_modularity, compute_conductance
from utils.graph_utils import preprocess_graph


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ScaleComm community detection results"
    )
    parser.add_argument("--dataset", type=str, default="cora",
        choices=["cora", "citeseer", "amazon-photo", "amazon-computers", "dblp"])
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Path to saved model checkpoint (.pt file)")
    parser.add_argument("--labels", type=str, default=None,
        help="Path to saved prediction labels (.npy file)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--cluster", type=str, default="kmeans",
        choices=["kmeans", "dpgmm"])
    parser.add_argument("--num_clusters", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--visualize", action="store_true",
        help="Generate community visualization plot")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_from_checkpoint(args):
    """Load model from checkpoint, re-run inference, then evaluate."""
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load dataset
    print(f"[Evaluate] Loading {args.dataset.upper()} dataset...")
    data = load_dataset(args.dataset)
    data = preprocess_graph(data)

    # Rebuild model
    model = GATEncoder(
        in_channels=data.num_node_features,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)

    # Load checkpoint
    print(f"[Evaluate] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Checkpoint from epoch {checkpoint['epoch']}, "
          f"loss {checkpoint['loss']:.4f}")

    # Extract embeddings
    print("[Evaluate] Extracting node embeddings...")
    embeddings = model.get_embeddings(
        data.x.to(device), data.edge_index.to(device)
    )

    # Cluster
    num_clusters = args.num_clusters or getattr(data, "num_classes", None)
    clusterer = get_clusterer(args.cluster, num_clusters=num_clusters)
    pred_labels, K = clusterer.fit_predict(embeddings)
    print(f"[Evaluate] Detected {K} communities")

    return data, pred_labels


def evaluate_from_labels(args):
    """Load pre-saved labels and evaluate against dataset ground truth."""
    print(f"[Evaluate] Loading {args.dataset.upper()} dataset...")
    data = load_dataset(args.dataset)
    data = preprocess_graph(data)

    print(f"[Evaluate] Loading labels from {args.labels}...")
    pred_labels = np.load(args.labels)
    return data, pred_labels


# ---------------------------------------------------------------------------
# Community analysis
# ---------------------------------------------------------------------------

def community_size_analysis(pred_labels: np.ndarray):
    """Print statistics about detected community sizes."""
    unique, counts = np.unique(pred_labels, return_counts=True)
    K = len(unique)

    print("\n" + "-" * 50)
    print(f"  COMMUNITY SIZE ANALYSIS  ({K} communities)")
    print("-" * 50)
    print(f"  {'ID':>4}  {'Size':>8}  {'% of graph':>12}  {'Bar'}")
    print("-" * 50)

    N = len(pred_labels)
    for cid, count in zip(unique, counts):
        pct = count / N * 100
        bar = "█" * int(pct / 2)
        print(f"  {int(cid):>4}  {count:>8}  {pct:>10.1f}%  {bar}")

    print("-" * 50)
    print(f"  Min size: {counts.min():,}   Max size: {counts.max():,}   "
          f"Mean size: {counts.mean():.1f}   Std: {counts.std():.1f}")
    print("-" * 50 + "\n")


def print_comparison_table(results: dict, dataset: str, K: int):
    """Print a formatted comparison table of results."""
    print("\n" + "╔" + "═" * 52 + "╗")
    print(f"║  ScaleComm Results — {dataset.upper():>20s} (K={K})    ║")
    print("╠" + "═" * 52 + "╣")

    # Reference values from literature for comparison
    baselines = {
        "cora": {
            "NMI": ("DGI", 0.528), "ARI": ("DGI", 0.441),
            "ACC": ("DCRN", 0.726), "MODULARITY": ("Louvain", 0.780),
        },
        "citeseer": {
            "NMI": ("DGI", 0.431), "ARI": ("DGI", 0.380),
            "ACC": ("DCRN", 0.671), "MODULARITY": ("Louvain", 0.710),
        },
    }
    ref = baselines.get(dataset, {})

    metric_labels = {
        "nmi": "NMI", "ari": "ARI", "acc": "ACC", "f1": "F1 Score",
        "modularity": "Modularity (Q)", "conductance": "Conductance (↓)"
    }

    for key, label in metric_labels.items():
        val = results.get(key)
        if val is None:
            continue
        ref_info = ref.get(label.split()[0], None)
        if ref_info:
            ref_str = f"  [baseline: {ref_info[1]:.3f} ({ref_info[0]})]"
        else:
            ref_str = ""
        print(f"║  {label:<20s}  {val:>8.4f}  {ref_str:<20s}║")

    print("╚" + "═" * 52 + "╝")


# ---------------------------------------------------------------------------
# Loss curve plot
# ---------------------------------------------------------------------------

def plot_loss_curve(dataset: str, save: bool = True):
    """Plot training loss curve if history file exists."""
    loss_file = f"outputs/loss_history_{dataset}.npy"
    if not os.path.exists(loss_file):
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    loss = np.load(loss_file)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss) + 1), loss, color="#2E75B6", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Contrastive Loss", fontsize=12)
    plt.title(f"ScaleComm Training Loss — {dataset.upper()}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = f"outputs/loss_curve_{dataset}.png"
        plt.savefig(path, dpi=150)
        print(f"[Plot] Loss curve saved to {path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("\n" + "═" * 60)
    print("  ScaleComm — Evaluation Mode")
    print("═" * 60 + "\n")

    # Get data and predictions
    if args.checkpoint is not None:
        data, pred_labels = evaluate_from_checkpoint(args)
    elif args.labels is not None:
        data, pred_labels = evaluate_from_labels(args)
    else:
        print("[Error] Provide either --checkpoint or --labels path.")
        print("  Example:")
        print("  python training/evaluate.py --dataset cora "
              "--checkpoint checkpoints/scalecomm_cora_best.pt")
        sys.exit(1)

    K = int(pred_labels.max()) + 1

    # Community size analysis
    community_size_analysis(pred_labels)

    # Full metric evaluation
    print("[Evaluate] Computing all evaluation metrics...")
    if hasattr(data, "y") and data.y is not None:
        results = evaluate_all(
            true_labels=data.y.numpy(),
            pred_labels=pred_labels,
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            verbose=True,
        )
    else:
        Q    = compute_modularity(data.edge_index, pred_labels, data.num_nodes)
        cond = compute_conductance(data.edge_index, pred_labels, data.num_nodes)
        results = {"modularity": Q, "conductance": cond}
        print(f"  Modularity  : {Q:.4f}")
        print(f"  Conductance : {cond:.4f}")

    # Comparison table
    print_comparison_table(results, args.dataset, K)

    # Visualization
    if args.visualize:
        from utils.graph_utils import visualize_communities
        os.makedirs("outputs", exist_ok=True)
        visualize_communities(
            data=data,
            labels=pred_labels,
            title=f"ScaleComm Communities — {args.dataset.upper()} (K={K})",
            save_path=f"outputs/communities_{args.dataset}.png"
        )

    # Plot loss curve
    plot_loss_curve(args.dataset)

    print("\n  Evaluation complete.\n")


if __name__ == "__main__":
    main()
