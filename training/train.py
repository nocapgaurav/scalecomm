"""
train.py
========
Full ScaleComm training pipeline.

Pipeline:
  1. Load dataset (Cora / CiteSeer / Amazon / etc.)
  2. Preprocess graph
  3. Initialize GATEncoder
  4. Contrastive pre-training with graph augmentation (InfoNCE)
  5. Extract final node embeddings
  6. Cluster embeddings → community labels
  7. Evaluate against ground-truth labels
  8. Save model checkpoint

Usage:
  python training/train.py --dataset cora --epochs 200 --device cpu
  python training/train.py --dataset citeseer --epochs 300 --device cuda
  python training/train.py --dataset amazon-photo --epochs 200 --cluster dpgmm
"""

import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

# ---- Add project root to path ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

from data.datasets import load_dataset, dataset_summary
from models.gat_encoder import GATEncoder, model_summary
from models.contrastive_loss import GraphAugmentor, ScaleCommLoss
from models.clustering import get_clusterer
from utils.graph_utils import preprocess_graph, print_graph_stats
from utils.metrics import evaluate_all


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ScaleComm: Self-Supervised Community Detection"
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="cora",
        choices=["cora", "citeseer", "amazon-photo", "amazon-computers", "dblp"],
        help="Dataset to use for training")

    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=256,
        help="GAT hidden dimension (default: 256)")
    parser.add_argument("--out_dim", type=int, default=128,
        help="GAT output embedding dimension (default: 128)")
    parser.add_argument("--num_heads", type=int, default=8,
        help="Number of GAT attention heads (default: 8)")
    parser.add_argument("--num_layers", type=int, default=3,
        help="Number of GAT layers (default: 3)")
    parser.add_argument("--dropout", type=float, default=0.3,
        help="Dropout rate (default: 0.3)")

    # Training
    parser.add_argument("--epochs", type=int, default=200,
        help="Training epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=1e-3,
        help="Learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
        help="L2 weight decay (default: 1e-5)")
    parser.add_argument("--temperature", type=float, default=0.5,
        help="InfoNCE temperature τ (default: 0.5)")

    # Augmentation
    parser.add_argument("--p_edge", type=float, default=0.2,
        help="Edge drop probability for augmentation (default: 0.2)")
    parser.add_argument("--p_feat", type=float, default=0.2,
        help="Feature mask probability for augmentation (default: 0.2)")

    # Clustering
    parser.add_argument("--cluster", type=str, default="kmeans",
        choices=["kmeans", "dpgmm"],
        help="Clustering method: 'kmeans' (fast) or 'dpgmm' (auto-K)")
    parser.add_argument("--num_clusters", type=int, default=None,
        help="Number of clusters for KMeans. None = auto-estimate")

    # Misc
    parser.add_argument("--device", type=str, default="auto",
        help="Device: 'cpu', 'cuda', or 'auto' (default: auto)")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for reproducibility")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
        help="Directory to save model checkpoint")
    parser.add_argument("--log_every", type=int, default=10,
        help="Log training progress every N epochs")
    parser.add_argument("--eval_every", type=int, default=50,
        help="Run full evaluation every N epochs")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
            print("[Device] Using Apple MPS (Metal)")
        else:
            dev = torch.device("cpu")
            print("[Device] Using CPU")
    else:
        dev = torch.device(device_arg)
        print(f"[Device] Using: {device_arg}")
    return dev


# ---------------------------------------------------------------------------
# Set seeds
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: GATEncoder,
    data,
    augmentor: GraphAugmentor,
    loss_fn: ScaleCommLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Run one training epoch of contrastive learning."""
    model.train()

    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)

    # Generate two augmented views of the graph
    (x1, ei1), (x2, ei2) = augmentor.get_two_views(x, edge_index)

    # Forward pass through encoder for both views
    z1, z1_proj = model(x1, ei1, return_projection=True)
    z2, z2_proj = model(x2, ei2, return_projection=True)

    # Compute combined loss
    loss_dict = loss_fn(
        z1=z1, z2=z2,
        z1_proj=z1_proj, z2_proj=z2_proj,
        x_orig=x,
    )

    # Backprop
    optimizer.zero_grad()
    loss_dict["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {k: v if isinstance(v, float) else v.item()
            for k, v in loss_dict.items()}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    # ---- Setup ----
    set_seed(args.seed)
    device = get_device(args.device)

    print("\n" + "█" * 60)
    print("  ScaleComm — Self-Supervised Community Detection")
    print("█" * 60)
    print(f"\n  Dataset    : {args.dataset.upper()}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Clustering : {args.cluster.upper()}")
    print(f"  Device     : {device}\n")

    # ---- Load & preprocess dataset ----
    print("[Step 1/5] Loading dataset...")
    data = load_dataset(args.dataset)
    dataset_summary(data)

    data = preprocess_graph(data)
    print_graph_stats(data)

    # ---- Initialize model ----
    print("[Step 2/5] Initializing GAT encoder...")
    model = GATEncoder(
        in_channels=data.num_node_features,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    model_summary(model, in_channels=data.num_node_features)

    # ---- Loss & optimizer ----
    augmentor = GraphAugmentor(p_edge=args.p_edge, p_feat=args.p_feat)
    loss_fn   = ScaleCommLoss(temperature=args.temperature, lambda_rec=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # ---- Training ----
    print("[Step 3/5] Starting contrastive training...\n")
    print(f"{'Epoch':>8}  {'Total Loss':>12}  {'Contrastive':>13}  {'Recon':>8}  {'LR':>10}")
    print("-" * 60)

    best_loss = float("inf")
    history = {"loss": [], "contrastive": [], "reconstruction": []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        loss_dict = train_one_epoch(
            model=model,
            data=data,
            augmentor=augmentor,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        scheduler.step()

        # Track history
        history["loss"].append(loss_dict["total"])
        history["contrastive"].append(loss_dict["contrastive"])
        history["reconstruction"].append(loss_dict["reconstruction"])

        # Log progress
        if epoch % args.log_every == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  {epoch:>6}  {loss_dict['total']:>12.4f}  "
                  f"{loss_dict['contrastive']:>13.4f}  "
                  f"{loss_dict['reconstruction']:>8.4f}  {lr:>10.6f}")

        # Intermediate evaluation
        if epoch % args.eval_every == 0:
            print(f"\n  [Epoch {epoch}] Running intermediate evaluation...")
            _quick_eval(model, data, args, device)
            print()

        # Save best model
        if loss_dict["total"] < best_loss:
            best_loss = loss_dict["total"]
            _save_checkpoint(model, optimizer, epoch, loss_dict["total"],
                             args.save_dir, args.dataset)

    elapsed = time.time() - start_time
    print(f"\n[Training Complete] {args.epochs} epochs in {elapsed:.1f}s "
          f"({elapsed/args.epochs:.2f}s/epoch)")

    # ---- Extract final embeddings ----
    print("\n[Step 4/5] Extracting node embeddings...")
    embeddings = model.get_embeddings(
        data.x.to(device),
        data.edge_index.to(device)
    )
    print(f"  Embedding shape: {embeddings.shape}")

    # ---- Clustering ----
    print("\n[Step 5/5] Clustering embeddings...")
    num_clusters = args.num_clusters
    if num_clusters is None and hasattr(data, "num_classes"):
        # Use ground truth K for fair comparison (KMeans mode)
        if args.cluster == "kmeans":
            num_clusters = data.num_classes
            print(f"  Using ground-truth K = {num_clusters} for KMeans")

    clusterer = get_clusterer(
        method=args.cluster,
        num_clusters=num_clusters,
    )
    pred_labels, K_pred = clusterer.fit_predict(embeddings)
    print(f"  Predicted communities: {K_pred}")

    # ---- Final evaluation ----
    print("\n" + "═" * 60)
    print("  FINAL EVALUATION RESULTS")
    print("═" * 60)

    if hasattr(data, "y") and data.y is not None:
        results = evaluate_all(
            true_labels=data.y.numpy(),
            pred_labels=pred_labels,
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            verbose=True,
        )
    else:
        print("  No ground-truth labels available. Computing modularity only.")
        from utils.metrics import compute_modularity
        Q = compute_modularity(data.edge_index, pred_labels, data.num_nodes)
        print(f"  Modularity (Q): {Q:.4f}")
        results = {"modularity": Q}

    # ---- Save results ----
    _save_results(results, pred_labels, history, args)

    return model, embeddings, pred_labels, results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quick_eval(model, data, args, device):
    """Run a quick evaluation snapshot during training."""
    embeddings = model.get_embeddings(
        data.x.to(device), data.edge_index.to(device)
    )

    num_clusters = args.num_clusters
    if num_clusters is None and hasattr(data, "num_classes"):
        num_clusters = data.num_classes

    # Quick KMeans only for intermediate eval (fast)
    from models.clustering import KMeansClustering
    clusterer = KMeansClustering(num_clusters=num_clusters, n_init=3)
    pred_labels, K = clusterer.fit_predict(embeddings)

    if hasattr(data, "y") and data.y is not None:
        from utils.metrics import compute_nmi, compute_ari, compute_modularity
        nmi = compute_nmi(data.y.numpy(), pred_labels)
        ari = compute_ari(data.y.numpy(), pred_labels)
        Q   = compute_modularity(data.edge_index, pred_labels, data.num_nodes)
        print(f"  → NMI: {nmi:.4f} | ARI: {ari:.4f} | Modularity: {Q:.4f} | K={K}")
    model.train()


def _save_checkpoint(model, optimizer, epoch, loss, save_dir, dataset):
    """Save model checkpoint to disk."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"scalecomm_{dataset}_best.pt")
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def _save_results(results, pred_labels, history, args):
    """Save evaluation results and community labels to disk."""
    os.makedirs("outputs", exist_ok=True)

    # Save community labels
    np.save(f"outputs/pred_labels_{args.dataset}.npy", pred_labels)

    # Save training history
    np.save(f"outputs/loss_history_{args.dataset}.npy", np.array(history["loss"]))

    # Print summary
    print(f"\n  Results saved to outputs/")
    print(f"  Community labels : outputs/pred_labels_{args.dataset}.npy")
    print(f"  Loss history     : outputs/loss_history_{args.dataset}.npy\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    model, embeddings, pred_labels, results = train(args)
