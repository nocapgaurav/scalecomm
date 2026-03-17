"""
clustering.py
=============
Clustering module for ScaleComm community detection.

Provides two clustering approaches:
  1. KMeansClustering   — Fast K-means with optional K auto-estimation
                          via gap statistic or silhouette analysis
  2. DPGMMClustering    — Dirichlet Process Gaussian Mixture Model
                          (Variational Bayes DPGMM via sklearn)
                          Automatically infers the number of communities.

Both classes accept node embeddings as input and return:
  - Community labels   [N]
  - Estimated K (number of communities)

Usage in ScaleComm pipeline:
  embeddings = encoder.get_embeddings(x, edge_index)
  clusterer  = DPGMMClustering(max_components=20)
  labels, K  = clusterer.fit_predict(embeddings)
"""

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseClustering:
    """Abstract base for all clustering strategies."""

    def fit_predict(self, embeddings):
        """
        Fit the clustering model and return community labels.

        Parameters
        ----------
        embeddings : torch.Tensor or np.ndarray  [N, D]

        Returns
        -------
        labels : np.ndarray [N]  — Integer community labels
        K      : int             — Number of detected communities
        """
        raise NotImplementedError

    def _to_numpy(self, embeddings):
        """Convert embeddings to L2-normalized numpy array."""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        # L2 normalize for cosine-similarity-aware clustering
        return normalize(embeddings, norm="l2")


# ---------------------------------------------------------------------------
# K-Means Clustering
# ---------------------------------------------------------------------------

class KMeansClustering(BaseClustering):
    """
    K-Means clustering on node embeddings.

    Supports:
      - Fixed K (when num_clusters is given)
      - Automatic K via silhouette analysis (when num_clusters is None)

    Parameters
    ----------
    num_clusters : int or None — Number of clusters. If None, auto-estimates.
    k_min        : int         — Minimum K to test (auto mode)
    k_max        : int         — Maximum K to test (auto mode)
    n_init       : int         — Number of K-means restarts (default 10)
    random_state : int         — Seed for reproducibility
    """

    def __init__(
        self,
        num_clusters: int = None,
        k_min: int = 2,
        k_max: int = 20,
        n_init: int = 10,
        random_state: int = 42,
    ):
        self.num_clusters = num_clusters
        self.k_min = k_min
        self.k_max = k_max
        self.n_init = n_init
        self.random_state = random_state

    def fit_predict(self, embeddings):
        """
        Cluster node embeddings and return community labels.

        If num_clusters is None, automatically finds best K using
        silhouette score over [k_min, k_max].
        """
        X = self._to_numpy(embeddings)
        N = X.shape[0]

        if self.num_clusters is not None:
            K = self.num_clusters
        else:
            print("[Clustering] Auto-estimating K via silhouette analysis...")
            K = self._estimate_k_silhouette(X)
            print(f"[Clustering] Estimated K = {K}")

        # Use MiniBatchKMeans for large graphs (N > 20000)
        if N > 20000:
            kmeans = MiniBatchKMeans(
                n_clusters=K,
                n_init=self.n_init,
                batch_size=4096,
                random_state=self.random_state,
            )
        else:
            kmeans = KMeans(
                n_clusters=K,
                n_init=self.n_init,
                max_iter=500,
                random_state=self.random_state,
            )

        labels = kmeans.fit_predict(X)
        return labels, K

    def _estimate_k_silhouette(self, X: np.ndarray) -> int:
        """
        Estimate optimal K by maximizing silhouette score.

        Subsample up to 5000 points for speed on large graphs.
        """
        N = X.shape[0]
        sample_size = min(N, 5000)
        idx = np.random.choice(N, sample_size, replace=False)
        X_sample = X[idx]

        best_k     = self.k_min
        best_score = -1.0
        scores     = []

        k_range = range(self.k_min, min(self.k_max + 1, sample_size))

        for k in k_range:
            km = KMeans(n_clusters=k, n_init=5, max_iter=200, random_state=42)
            lbl = km.fit_predict(X_sample)

            if len(np.unique(lbl)) < 2:
                scores.append(-1.0)
                continue

            score = silhouette_score(X_sample, lbl, metric="cosine",
                                     sample_size=min(2000, sample_size))
            scores.append(score)

            if score > best_score:
                best_score = score
                best_k = k

        print(f"[Clustering] Silhouette scores tested K={list(k_range)}")
        return best_k


# ---------------------------------------------------------------------------
# DPGMM Clustering (Automatic K)
# ---------------------------------------------------------------------------

class DPGMMClustering(BaseClustering):
    """
    Dirichlet Process Gaussian Mixture Model clustering.

    Uses Variational Bayesian inference (sklearn BayesianGaussianMixture)
    with a Dirichlet Process prior. The model automatically determines
    the effective number of communities by "turning off" unused components.

    Parameters
    ----------
    max_components  : int   — Upper bound on number of communities (default 30)
    covariance_type : str   — 'full', 'tied', 'diag', or 'spherical'
    n_init          : int   — Number of restarts (default 3)
    max_iter        : int   — Max EM iterations (default 300)
    weight_threshold: float — Components with weight below this are ignored
    random_state    : int   — Seed for reproducibility
    """

    def __init__(
        self,
        max_components: int = 30,
        covariance_type: str = "diag",
        n_init: int = 3,
        max_iter: int = 300,
        weight_threshold: float = 0.01,
        random_state: int = 42,
    ):
        self.max_components   = max_components
        self.covariance_type  = covariance_type
        self.n_init           = n_init
        self.max_iter         = max_iter
        self.weight_threshold = weight_threshold
        self.random_state     = random_state
        self.model            = None

    def fit_predict(self, embeddings):
        """
        Fit DPGMM and return community labels with auto-estimated K.

        Returns
        -------
        labels : np.ndarray [N] — Remapped contiguous community labels
        K      : int            — Number of active components
        """
        X = self._to_numpy(embeddings)

        # Reduce dimensionality for speed if embedding dim > 64
        if X.shape[1] > 64:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=64, random_state=self.random_state)
            X = pca.fit_transform(X)
            print(f"[Clustering] Applied PCA: {embeddings.shape[1]}→64 dims")

        print(f"[Clustering] Fitting DPGMM (max_components={self.max_components})...")

        self.model = BayesianGaussianMixture(
            n_components=self.max_components,
            covariance_type=self.covariance_type,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=1.0 / self.max_components,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0,
        )

        raw_labels = self.model.fit_predict(X)

        # Count active components (those above weight threshold)
        active_weights = self.model.weights_ > self.weight_threshold
        active_idx = np.where(active_weights)[0]
        K = int(active_idx.sum())

        if K == 0:
            # Fallback: at least 2 communities
            K = 2
            active_idx = np.argsort(self.model.weights_)[-2:]

        print(f"[Clustering] DPGMM found K={K} active communities")
        print(f"  Component weights: {np.round(self.model.weights_[active_idx], 3)}")

        # Remap labels to contiguous integers [0, K)
        label_map = {old: new for new, old in enumerate(active_idx)}
        # Nodes assigned to inactive components → nearest active component
        labels = self._remap_labels(raw_labels, label_map, X)

        return labels, K

    def _remap_labels(self, raw_labels, label_map, X):
        """Remap raw labels to contiguous range, reassigning inactive clusters."""
        N = len(raw_labels)
        new_labels = np.full(N, -1, dtype=int)

        # First pass: map active labels directly
        for i in range(N):
            if raw_labels[i] in label_map:
                new_labels[i] = label_map[raw_labels[i]]

        # Second pass: reassign unassigned nodes to nearest active centroid
        unassigned = np.where(new_labels == -1)[0]
        if len(unassigned) > 0:
            active_centers = np.array(list(label_map.keys()))
            means = self.model.means_[active_centers]  # [K, D]
            X_unassigned = X[unassigned]               # [M, D]
            # Compute distances to active centroids
            dists = np.sum(
                (X_unassigned[:, None, :] - means[None, :, :]) ** 2,
                axis=-1
            )  # [M, K]
            nearest = np.argmin(dists, axis=1)
            for idx, node in enumerate(unassigned):
                new_labels[node] = nearest[idx]

        return new_labels


# ---------------------------------------------------------------------------
# Clustering factory
# ---------------------------------------------------------------------------

def get_clusterer(method: str = "dpgmm", num_clusters: int = None, **kwargs):
    """
    Factory function to instantiate the appropriate clusterer.

    Parameters
    ----------
    method       : 'kmeans' or 'dpgmm'
    num_clusters : For kmeans — number of communities (None = auto-estimate)
    **kwargs     : Additional arguments passed to the clusterer

    Returns
    -------
    clusterer : BaseClustering instance
    """
    method = method.lower()

    if method == "kmeans":
        return KMeansClustering(num_clusters=num_clusters, **kwargs)
    elif method == "dpgmm":
        return DPGMMClustering(**kwargs)
    else:
        raise ValueError(f"Unknown clustering method '{method}'. "
                         "Choose 'kmeans' or 'dpgmm'.")
