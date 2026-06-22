"""Feature extraction backbone for the score-based denoiser.

The paper (Sec. 4.1) implements the feature-extraction module ``E`` as a
DGCNN [Wang et al. 2019]: a stack of dynamic edge-convolutions that re-build the
k-NN graph in *feature* space at every layer and aggregate neighbour context.

This is a self-contained port (the ``sdn`` package does not depend on ``vm`` /
``asdn``), mirroring the DGCNN used by the score-denoise family of methods.
"""

from typing import Optional

import jittor as jt
from jittor import nn


# --------------------------------------------------------------------------- #
# k-NN helper (works for arbitrary feature dimension via a random 3D projection
# for the feature-space graph, exactly like the existing vm/asdn backbones)
# --------------------------------------------------------------------------- #
_knn_proj_cache = {}


def _get_knn_proj(in_dim):
    if in_dim not in _knn_proj_cache:
        proj = jt.randn(in_dim, 3)
        proj = proj / jt.norm(proj, dim=0, keepdims=True)
        _knn_proj_cache[in_dim] = proj
    return _knn_proj_cache[in_dim]


def get_knn_idx(x, y, k, offset=0):
    """k nearest neighbours of every point in ``x`` among points in ``y``.

    x: (B, N, d) ; y: (B, M, d) -> idx: (B, N, k)
    """
    K = k + offset
    if x.shape[-1] == 3:
        _, idx = jt.misc.knn(x, y, K)
    else:
        proj = _get_knn_proj(x.shape[-1])
        x_proj = x @ proj
        y_proj = y @ proj
        _, idx = jt.misc.knn(x_proj, y_proj, K)
    return idx[:, :, offset:]


class EdgeConv(nn.Module):
    """A single DGCNN edge-convolution operating on a flattened graph."""

    def __init__(self, in_channels, out_channels, activation: Optional[str] = 'ReLU'):
        super().__init__()
        if activation == 'ReLU':
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels), nn.ReLU(),
                nn.Linear(out_channels, out_channels), nn.ReLU(),
            )
            self.lin = nn.Sequential(nn.Linear(in_channels, out_channels), nn.ReLU())
        elif activation is None:
            self.mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels), nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.lin = nn.Linear(in_channels, out_channels)
        else:
            raise Exception("Please assign a valid activation to EdgeConv!")

    def execute(self, x, edge_index):
        # x: (V, C), edge_index: (2, E) with row 0 = source, row 1 = destination
        src, dst = edge_index[0], edge_index[1]
        x_i = x[dst]               # (E, C) center
        x_j = x[src]               # (E, C) neighbour
        msg = self.mlp(jt.concat([x_i, x_j - x_i], dim=1))   # edge feature

        V = x.shape[0]
        out = jt.full((V, msg.shape[1]), 0)
        cnt = jt.full((V, msg.shape[1]), 0)
        out = out.scatter_(0, dst.unsqueeze(1).broadcast(msg.shape), msg, reduce='add')
        cnt = cnt.scatter_(0, dst.unsqueeze(1).broadcast(msg.shape), jt.ones_like(msg), reduce='add')
        out = out / (cnt + 1)
        return out + self.lin(x)


class FeatureExtraction(nn.Module):
    """DGCNN feature extractor ``E`` (paper Sec. 4.1).

    Input  : point cloud (B, N, 3)
    Output : per-point features (B, N, embedding_dim)
    """

    def __init__(self, k=32, input_dim=3, embedding_dim=128):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.conv1 = EdgeConv(input_dim, embedding_dim // 8)
        self.conv2 = EdgeConv(embedding_dim // 8, embedding_dim // 4)
        self.conv3 = EdgeConv(embedding_dim // 8 + embedding_dim // 4,
                              embedding_dim, activation=None)

    def get_edge_index(self, x):
        # Build a directed k-NN graph (excluding self) for batched point clouds.
        B, N, _ = x.shape
        knn_idx = get_knn_idx(x, x, self.k + 1)[:, :, 1:]    # drop self -> (B, N, k)
        base = (jt.arange(B) * N).reshape(B, 1, 1)
        src = (knn_idx + base).reshape(-1)
        dst = (jt.arange(N).reshape(1, N, 1).broadcast((B, N, self.k)) + base).reshape(-1)
        return jt.stack([src, dst], dim=0)

    def execute(self, x):
        B, N, _ = x.shape

        edge_index = self.get_edge_index(x)
        x1 = self.conv1(x.reshape(B * N, -1), edge_index).reshape(B, N, -1)

        edge_index = self.get_edge_index(x1)
        x2 = self.conv2(x1.reshape(B * N, -1), edge_index).reshape(B, N, -1)

        edge_index = self.get_edge_index(x2)
        x_cat = jt.concat([x1, x2], dim=-1)
        x3 = self.conv3(x_cat.reshape(B * N, -1), edge_index).reshape(B, N, -1)
        return x3
