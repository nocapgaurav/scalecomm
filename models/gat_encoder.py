"""
gat_encoder.py
==============
Implements the Graph Attention Network (GAT) encoder used in ScaleComm.

Architecture:
  - 3-layer GAT with multi-head attention
  - ELU activations + dropout regularization
  - Layer normalization for training stability
  - Final projection head for contrastive learning

The encoder produces node embeddings that capture both:
  1. Graph topology (via attention-weighted neighborhood aggregation)
  2. Node attribute information (via feature transformation)

Reference:
  Velickovic et al., "Graph Attention Networks", ICLR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm


# ---------------------------------------------------------------------------
# GAT Encoder
# ---------------------------------------------------------------------------

class GATEncoder(nn.Module):
    """
    Multi-layer Graph Attention Network encoder.

    Produces node embeddings suitable for contrastive learning
    and downstream community detection.

    Parameters
    ----------
    in_channels   : int   — Input feature dimension
    hidden_dim    : int   — Hidden embedding dimension (default 256)
    out_dim       : int   — Output embedding dimension (default 128)
    num_heads     : int   — Number of attention heads per layer (default 8)
    num_layers    : int   — Number of GAT layers (default 3)
    dropout       : float — Dropout probability (default 0.3)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super(GATEncoder, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # ----- Build GAT layers -----
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # First layer: in_channels → hidden_dim
                conv = GATConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,       # concat multi-head outputs
                )
            elif i == num_layers - 1:
                # Last layer: hidden_dim → out_dim (single head, no concat)
                conv = GATConv(
                    in_channels=hidden_dim,
                    out_channels=out_dim,
                    heads=1,
                    dropout=dropout,
                    concat=False,      # average multi-head outputs
                )
            else:
                # Middle layers: hidden_dim → hidden_dim
                conv = GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )

            self.convs.append(conv)
            # Batch norm after each layer except the last
            if i < num_layers - 1:
                self.norms.append(BatchNorm(hidden_dim))

        # ----- Projection head for contrastive learning -----
        # Maps from out_dim → proj_dim (used only during contrastive training)
        self.projection_head = ProjectionHead(
            in_dim=out_dim,
            hidden_dim=out_dim * 2,
            out_dim=out_dim,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                return_projection: bool = False):
        """
        Forward pass through the GAT encoder.

        Parameters
        ----------
        x               : Node feature matrix [N, F]
        edge_index      : Graph connectivity [2, E]
        return_projection: If True, also returns the projection head output
                          (used during contrastive training)

        Returns
        -------
        z      : Node embeddings from final GAT layer [N, out_dim]
        z_proj : (optional) Projected embeddings [N, out_dim]
        """
        h = x

        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)

            if i < self.num_layers - 1:
                # Apply norm + activation + dropout between layers
                h = self.norms[i](h)
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        # h is now the final node embedding z ∈ R^[N, out_dim]
        z = h

        if return_projection:
            z_proj = self.projection_head(z)
            return z, z_proj

        return z

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Get node embeddings in inference mode (no dropout).
        Use this for clustering / evaluation.
        """
        self.eval()
        with torch.no_grad():
            z = self.forward(x, edge_index, return_projection=False)
        return z

    def reset_parameters(self):
        """Re-initialize all layer parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        self.projection_head.reset_parameters()


# ---------------------------------------------------------------------------
# Projection Head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head used during contrastive training.

    Maps encoder output embeddings to a new space where the
    contrastive loss is computed (standard practice from SimCLR).
    The projection head is discarded at inference time; only the
    GATEncoder output is used for clustering.

    Parameters
    ----------
    in_dim    : int — Input dimension (= encoder out_dim)
    hidden_dim: int — Hidden layer size
    out_dim   : int — Output dimension for contrastive loss
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super(ProjectionHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def reset_parameters(self):
        for layer in self.net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


# ---------------------------------------------------------------------------
# Model summary utility
# ---------------------------------------------------------------------------

def model_summary(model: GATEncoder, in_channels: int, device: str = "cpu"):
    """Print a summary of the model architecture and parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 50)
    print("GAT ENCODER ARCHITECTURE")
    print("=" * 50)
    print(model)
    print("-" * 50)
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print("=" * 50 + "\n")
