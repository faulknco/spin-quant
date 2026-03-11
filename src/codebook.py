"""
Codebook construction via k-means.

Physics framing:
  - Each weight block is a "site" on a lattice.
  - The codebook C is a finite alphabet (like discrete spin states).
  - Quantization = assigning each site its nearest spin state.
"""

import torch


def kmeans(x: torch.Tensor, K: int, n_iter: int = 50, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Lloyd's k-means on CPU.

    Args:
        x:      [N, d] float tensor of data points (weight blocks)
        K:      number of centroids (codebook size)
        n_iter: number of Lloyd iterations
        seed:   random seed for centroid initialization

    Returns:
        centroids: [K, d]
        labels:    [N] long, centroid index per block
    """
    torch.manual_seed(seed)
    N, d = x.shape
    assert K <= N, f"K={K} must be <= N={N}"

    # kmeans++ style: pick K random points as initial centroids
    perm = torch.randperm(N)[:K]
    centroids = x[perm].clone()

    labels = torch.zeros(N, dtype=torch.long)

    for _ in range(n_iter):
        # [N, K] squared distances
        dist = torch.cdist(x, centroids, p=2).pow(2)
        labels = dist.argmin(dim=1)

        # update centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(K, dtype=torch.long)
        new_centroids.scatter_add_(0, labels.unsqueeze(1).expand(-1, d), x)
        counts.scatter_add_(0, labels, torch.ones(N, dtype=torch.long))

        # handle empty clusters: keep old centroid
        nonempty = counts > 0
        new_centroids[nonempty] /= counts[nonempty].float().unsqueeze(1)
        new_centroids[~nonempty] = centroids[~nonempty]
        centroids = new_centroids

    return centroids, labels


def quantize_blocks(
    W: torch.Tensor,
    block_dim: int,
    K: int,
    n_iter: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reshape W into blocks and run k-means codebook quantization.

    Args:
        W:         [out_features, in_features]
        block_dim: size d of each block vector (must divide in_features)
        K:         codebook size

    Returns:
        centroids: [K, block_dim]
        labels:    [out_features * (in_features // block_dim)]  -- one index per block
        W_shape:   original W.shape (for reconstruction)
    """
    out_f, in_f = W.shape
    assert in_f % block_dim == 0, f"in_features={in_f} must be divisible by block_dim={block_dim}"

    W_blocks = W.reshape(-1, block_dim).float().cpu()
    centroids, labels = kmeans(W_blocks, K, n_iter=n_iter)
    return centroids, labels, W.shape


def reconstruct(centroids: torch.Tensor, labels: torch.Tensor, W_shape: tuple) -> torch.Tensor:
    """Dequantize: map indices back to centroid vectors and reshape."""
    W_blocks = centroids[labels]
    return W_blocks.view(W_shape)


def quantization_rmse(W: torch.Tensor, centroids: torch.Tensor, labels: torch.Tensor) -> float:
    Wq = reconstruct(centroids, labels, W.shape)
    return (W.float().cpu() - Wq).pow(2).mean().sqrt().item()
