"""Graph-convolution feature extraction for the PD-Flow denoiser.

Faithful Jittor port of the EdgeConv-style layers from PD-LTS / PD-Flow
(yanbiao1/PD-LTS, models/model_light/layer.py). These produce the per-point
conditioning features that are injected into the normalizing flow.
"""

import jittor as jt
from jittor import nn


def get_knn_idx(x, k):
    """k nearest neighbours (including self) of every point. x:(B,N,C)->idx:(B,N,k)."""
    _, idx = jt.misc.knn(x, x, k)
    return idx


def knn_gather(f, idx):
    """Gather neighbour features. f:(B,N,C), idx:(B,M,k) -> (B,M,k,C).

    Vectorised (single gather, no Python loop over the batch) -- the per-batch
    loop was the main speed bottleneck.
    """
    B, N, C = f.shape
    M, k = idx.shape[1], idx.shape[2]
    base = (jt.arange(B) * N).reshape(B, 1, 1)
    idx_flat = (idx + base).reshape(-1)                # (B*M*k,)
    return f.reshape(B * N, C)[idx_flat].reshape(B, M, k, C)


class noiseEdgeConv(nn.Module):
    """Embeds local geometry into the augmentation channels (aug feature)."""

    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.linear1 = nn.Linear(in_channel * 2, hidden_channel)
        self.linear2 = nn.Linear(hidden_channel, hidden_channel)
        self.linear3 = nn.Linear(in_channel, hidden_channel)
        self.linear4 = nn.Linear(hidden_channel, hidden_channel)
        self.linear5 = nn.Linear(hidden_channel, out_channel)
        nn.init.gauss_(self.linear5.weight, 0.0, 0.05)
        nn.init.constant_(self.linear5.bias, 0.0)

    def execute(self, f, knn_idx):
        knn_feat = knn_gather(f, knn_idx)                       # (B,N,k,C)
        f_tiled = f.unsqueeze(2).broadcast(knn_feat.shape)
        x = jt.concat([knn_feat, knn_feat - f_tiled], dim=-1)   # (B,N,k,2C)
        x = nn.relu(self.linear1(x))
        x = nn.relu(self.linear2(x))
        x = jt.max(x, dim=2)                                    # (B,N,h)
        g = nn.relu(self.linear3(f))
        g = nn.relu(self.linear4(g))
        x = x + g
        return self.linear5(x)


class PreConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel * 2, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.05),
        )

    def execute(self, f, knn_idx):
        knn_feat = knn_gather(f, knn_idx)
        f_tiled = f.unsqueeze(2).broadcast(knn_feat.shape)
        x = jt.concat([f_tiled, knn_feat - f_tiled], dim=-1)    # (B,N,k,2C)
        x = x.permute(0, 3, 1, 2)                               # (B,2C,N,k)
        x = self.conv(x)
        x = jt.max(x, dim=-1)                                   # (B,out,N)
        return x.transpose(1, 2)                                # (B,N,out)


class EdgeConv(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, concat=True):
        super().__init__()
        self.concat = concat
        if not concat:
            hidden_channel = hidden_channel + 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel * 2, hidden_channel, 1),
            nn.BatchNorm2d(hidden_channel), nn.LeakyReLU(0.05))
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel), nn.LeakyReLU(0.05))

    def execute(self, f, knn_idx):
        knn_feat = knn_gather(f, knn_idx)
        f_tiled = f.unsqueeze(2).broadcast(knn_feat.shape)
        x = jt.concat([f_tiled, knn_feat - f_tiled], dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.conv2(self.conv1(x))
        x = jt.max(x, dim=-1)
        x = x.transpose(1, 2)
        if self.concat:
            x = jt.concat([x, f], dim=-1)
        return x


class FeatMergeUnit(nn.Module):
    """1x1 conv merge -> injector feature of dim (pc_channel + aug_channel)."""

    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, hidden_channel, 1),
            nn.BatchNorm1d(hidden_channel), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_channel, out_channel, 1),
            nn.BatchNorm1d(out_channel), nn.ReLU())

    def execute(self, x):
        x = x.transpose(1, 2)
        x = self.conv2(self.conv1(x))
        return x.transpose(1, 2)
