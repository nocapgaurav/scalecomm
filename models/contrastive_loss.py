"""
contrastive_loss.py
===================
Implements self-supervised contrastive loss for graph node embeddings.

Two components:
  1. GraphAugmentor  — Creates two augmented views of the input graph
                       using community-aware perturbations:
                         • Edge dropping (preserves high-degree edges)
                         • Feature masking (random feature zeroing)

  2. InfoNCELoss     — Node-level contrastive loss (NT-Xent / InfoNCE)
                       Positive pair: same node in the two augmented views
                       Negative pairs: all other nodes in the batch

This follows the GRACE-CD design but with community-aware augmentation:
high-betweenness edges (likely inter-community) are dropped with higher
probability, preserving intra-community structure in the augmented views.

References:
  - Chen et al., "A Simple Framework for Contrastive Learning", ICML 2020
  - Zhu et al., "Deep Graph Contrastive Representation Learning", 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Graph Augmentor
# ---------------------------------------------------------------------------

class GraphAugmentor:
    """
    Produces two stochastically augmented views of a graph.

    Augmentation strategies:
      1. Edge dropping   : Remove edges with probability p_edge
                           (edges with lower degree sum dropped preferentially)
      2. Feature masking : Zero out node features with probability p_feat

    Parameters
    ----------
    p_edge : float — Edge drop probability (default 0.2)
    p_feat : float — Feature mask probability (default 0.2)
    """

    def __init__(self, p_edge: float = 0.2, p_feat: float = 0.2):
        self.p_edge = p_edge
        self.p_feat = p_feat

    def augment(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Apply both augmentations to produce a single augmented view.

        Returns
        -------
        x_aug        : Augmented feature matrix [N, F]
        edge_index_aug: Augmented edge index [2, E']
        """
        x_aug = self._mask_features(x)
        edge_index_aug = self._drop_edges(edge_index, x.size(0))
        return x_aug, edge_index_aug

    def get_two_views(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Produce two independent augmented views of the same graph.

        Returns
        -------
        (x1, edge1) : View 1
        (x2, edge2) : View 2
        """
        view1 = self.augment(x, edge_index)
        view2 = self.augment(x, edge_index)
        return view1, view2

    def _drop_edges(self, edge_index: torch.Tensor, num_nodes: int):
        """
        Randomly remove edges.

        Edges connecting low-degree nodes (likely inter-community bridges)
        are given slightly higher drop probability — this is the
        'community-aware' aspect that preserves intra-community structure.
        """
        num_edges = edge_index.size(1)

        # Compute node degrees for degree-weighted dropping
        row = edge_index[0]
        degree = torch.bincount(row, minlength=num_nodes).float()
        degree = degree / (degree.max() + 1e-8)  # normalize to [0, 1]

        # Edge-level drop probability: lower degree → higher drop prob
        edge_degree = degree[row]
        # Invert: low-degree edges drop with up to 2× base probability
        drop_prob = self.p_edge * (2.0 - edge_degree)
        drop_prob = drop_prob.clamp(0.0, 0.95)  # cap at 95%

        # Sample a Bernoulli mask: 1 = keep, 0 = drop
        keep_mask = torch.bernoulli(1.0 - drop_prob).bool()

        return edge_index[:, keep_mask]

    def _mask_features(self, x: torch.Tensor):
        """
        Randomly zero out individual node features.
        Each feature dimension of each node is masked independently.
        """
        # Bernoulli mask: 1 = keep, 0 = zero out
        mask = torch.bernoulli(
            torch.full(x.shape, 1.0 - self.p_feat, device=x.device)
        )
        return x * mask


# ---------------------------------------------------------------------------
# InfoNCE Contrastive Loss
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """
    Node-level InfoNCE (NT-Xent) contrastive loss.

    Given two views of the same graph:
      - z1 [N, D]: embeddings from view 1 (L2-normalized)
      - z2 [N, D]: embeddings from view 2 (L2-normalized)

    For each node i:
      - Positive pair: (z1_i, z2_i) — same node in both views
      - Negative pairs: (z1_i, z2_j) for all j ≠ i, and vice versa

    Loss = mean of cross-entropy over all 2N anchor nodes.

    Parameters
    ----------
    temperature : float — Softmax temperature τ (default 0.5)
    """

    def __init__(self, temperature: float = 0.5):
        super(InfoNCELoss, self).__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss between two sets of embeddings.

        Parameters
        ----------
        z1 : Embeddings from view 1 [N, D]
        z2 : Embeddings from view 2 [N, D]

        Returns
        -------
        loss : Scalar contrastive loss
        """
        N = z1.size(0)
        device = z1.device

        # L2-normalize embeddings to lie on unit hypersphere
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate both views: [2N, D]
        z = torch.cat([z1, z2], dim=0)

        # Similarity matrix [2N, 2N]: cosine similarities scaled by τ
        sim = torch.mm(z, z.T) / self.tau  # [2N, 2N]

        # Mask out self-similarities (diagonal) with large negative value
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, -1e9)

        # Positive pair indices:
        # For anchor i in [0, N): positive is i + N
        # For anchor i in [N, 2N): positive is i - N
        pos_idx = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device),
        ])  # [2N]

        # Cross-entropy loss: treat positive as the target class
        loss = F.cross_entropy(sim, pos_idx)

        return loss


# ---------------------------------------------------------------------------
# Combined ScaleComm training loss
# ---------------------------------------------------------------------------

class ScaleCommLoss(nn.Module):
    """
    Combined loss for ScaleComm training.

    L_total = L_contrastive + λ * L_reconstruction

    where:
      L_contrastive : InfoNCE loss between two augmented views
      L_reconstruction: MSE reconstruction loss from the encoder
                        (regularization to preserve graph structure)

    Parameters
    ----------
    temperature : float — InfoNCE temperature (default 0.5)
    lambda_rec  : float — Weight for reconstruction term (default 0.1)
    """

    def __init__(self, temperature: float = 0.5, lambda_rec: float = 0.1):
        super(ScaleCommLoss, self).__init__()
        self.contrastive = InfoNCELoss(temperature)
        self.lambda_rec  = lambda_rec

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z1_proj: torch.Tensor = None,
        z2_proj: torch.Tensor = None,
        x_orig: torch.Tensor = None,
    ) -> dict:
        """
        Compute total ScaleComm loss.

        Parameters
        ----------
        z1, z2          : Encoder output embeddings from view 1 and view 2
        z1_proj, z2_proj: Projection head outputs (used for contrastive loss)
        x_orig          : Original node features (for reconstruction term)

        Returns
        -------
        result : dict with keys 'total', 'contrastive', 'reconstruction'
        """
        # Use projection head outputs for contrastive loss if available
        c1 = z1_proj if z1_proj is not None else z1
        c2 = z2_proj if z2_proj is not None else z2

        # InfoNCE contrastive loss
        l_con = self.contrastive(c1, c2)

        # Optional reconstruction regularization
        l_rec = torch.tensor(0.0, device=z1.device)
        if x_orig is not None and self.lambda_rec > 0:
            # Mean of embeddings from both views as reconstruction target
            z_avg = (z1 + z2) / 2.0
            # Project back to original feature space (simple linear approx)
            l_rec = F.mse_loss(
                F.normalize(z_avg, dim=1),
                F.normalize(x_orig[:, :z_avg.size(1)], dim=1)
                if x_orig.size(1) >= z_avg.size(1)
                else F.pad(x_orig, (0, z_avg.size(1) - x_orig.size(1)))
            )

        total = l_con + self.lambda_rec * l_rec

        return {
            "total": total,
            "contrastive": l_con.item(),
            "reconstruction": l_rec.item(),
        }
