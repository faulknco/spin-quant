"""
Drop-in nn.Linear replacements that store weights as codebook indices.

Usage:
    layer = CodebookLinear.from_linear(original_linear, block_dim=16, K=256)
    model.transformer.h[0].mlp.c_fc = layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook import quantize_blocks, reconstruct
from .physics import on_quantize, on_reconstruct, rg_quantize, rg_reconstruct


class CodebookLinear(nn.Module):
    """
    Linear layer with weights compressed via scalar vector codebook.

    Storage: centroids [K, d] + labels [N_blocks] (int8/int16) instead of full weight matrix.
    """

    def __init__(self, in_features, out_features, block_dim, centroids, labels, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_dim = block_dim
        self.register_buffer("centroids", centroids.float())
        self.register_buffer("labels", labels)
        if bias is not None:
            self.bias = nn.Parameter(bias.clone().float())
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_dim: int = 16, K: int = 256, n_iter: int = 50):
        W = linear.weight.data
        centroids, labels, _ = quantize_blocks(W, block_dim, K, n_iter=n_iter)
        bias = linear.bias.data if linear.bias is not None else None
        return cls(linear.in_features, linear.out_features, block_dim, centroids, labels, bias)

    def dequantize(self) -> torch.Tensor:
        return reconstruct(self.centroids, self.labels, (self.out_features, self.in_features))

    def forward(self, x):
        W = self.dequantize()
        return F.linear(x, W, self.bias)

    def bits_per_weight(self) -> float:
        import math
        K = self.centroids.shape[0]
        return math.log2(K) / self.block_dim


class ONLinear(nn.Module):
    """
    O(N)-style: direction codebook + discrete norm ladder.
    Physics framing: spin orientation (unit vector on S^{d-1}) + scalar magnitude.
    """

    def __init__(self, in_features, out_features, state: dict, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("dir_centroids", state["dir_centroids"].float())
        self.register_buffer("dir_labels", state["dir_labels"])
        self.register_buffer("norm_bins", state["norm_bins"].float())
        self.register_buffer("norm_idx", state["norm_idx"])
        self._W_shape = state["W_shape"]
        self._block_dim = state["block_dim"]
        if bias is not None:
            self.bias = nn.Parameter(bias.clone().float())
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_dim: int = 16, K_dir: int = 256,
                    n_levels: int = 16, n_iter: int = 50):
        state = on_quantize(linear.weight.data, block_dim, K_dir, n_levels, n_iter)
        bias = linear.bias.data if linear.bias is not None else None
        return cls(linear.in_features, linear.out_features, state, bias)

    def dequantize(self) -> torch.Tensor:
        state = {
            "dir_centroids": self.dir_centroids,
            "dir_labels": self.dir_labels,
            "norm_bins": self.norm_bins,
            "norm_idx": self.norm_idx,
            "W_shape": self._W_shape,
        }
        return on_reconstruct(state)

    def forward(self, x):
        return F.linear(x, self.dequantize(), self.bias)


class RGLinear(nn.Module):
    """
    Hierarchical (RG-style) codebook: coarse + residual correction.
    Physics framing: coarse = long-wavelength modes, residual = short-wavelength corrections.
    """

    def __init__(self, in_features, out_features, state: dict, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("coarse_centroids", state["coarse_centroids"].float())
        self.register_buffer("coarse_labels", state["coarse_labels"])
        self.register_buffer("res_centroids", state["res_centroids"].float())
        self.register_buffer("res_labels", state["res_labels"])
        self._W_shape = state["W_shape"]
        self._block_dim = state["block_dim"]
        if bias is not None:
            self.bias = nn.Parameter(bias.clone().float())
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_dim: int = 16,
                    K_coarse: int = 64, K_residual: int = 64, n_iter: int = 50):
        state = rg_quantize(linear.weight.data, block_dim, K_coarse, K_residual, n_iter)
        bias = linear.bias.data if linear.bias is not None else None
        return cls(linear.in_features, linear.out_features, state, bias)

    def dequantize(self) -> torch.Tensor:
        state = {
            "coarse_centroids": self.coarse_centroids,
            "coarse_labels": self.coarse_labels,
            "res_centroids": self.res_centroids,
            "res_labels": self.res_labels,
            "W_shape": self._W_shape,
        }
        return rg_reconstruct(state)

    def forward(self, x):
        return F.linear(x, self.dequantize(), self.bias)
