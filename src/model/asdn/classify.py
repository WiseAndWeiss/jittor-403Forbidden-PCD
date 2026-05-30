import math
import jittor as jt
from jittor import nn

from ..feature import get_knn_idx


def remove_self_loops(edge_index):
    src, dst = edge_index[0], edge_index[1]
    mask = src != dst
    return edge_index[:, mask]


class EdgeConv(nn.Module):
    """EdgeConv with mean aggregation (official uses max, but Jittor scatter
    maximum lacks gradient support; mean works and the network adapts)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.lin = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def execute(self, x, edge_index):
        src = edge_index[0]
        dst = edge_index[1]

        x_i = x[dst]
        x_j = x[src]

        tmp = jt.concat([x_i, x_j - x_i], dim=1)
        msg = self.mlp(tmp)

        N = x.shape[0]
        out = jt.zeros((N, msg.shape[1]))
        cnt = jt.zeros((N, msg.shape[1]))
        out = out.scatter_(0, dst.unsqueeze(1).broadcast(msg.shape), msg, reduce='add')
        cnt = cnt.scatter_(0, dst.unsqueeze(1).broadcast(msg.shape), jt.ones_like(msg), reduce='add')
        out = out / (cnt + 1)

        out_2 = self.lin(x)
        return out + out_2


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)

    def execute(self, x, edge_index):
        return super().execute(x, edge_index)


class ScaleNet(nn.Module):
    """Classifier network that predicts entropy ratio for each point cloud.
    Architecture: 4 DynamicEdgeConv -> global mean pool -> MLP -> tanh -> rho"""

    def __init__(self, k=32, input_dim=0, z_dim=0, embedding_dim=512, output_dim=1):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        self.conv1 = DynamicEdgeConv(3, 16)
        self.conv2 = DynamicEdgeConv(16, 24)
        self.conv3 = DynamicEdgeConv(24, 72)
        self.conv4 = DynamicEdgeConv(16 + 24 + 72, self.embedding_dim)

        self.linear1 = nn.Linear(self.embedding_dim, 128, bias=False)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, self.output_dim)

        if self.z_dim > 0:
            self.linear_proj = nn.Linear(256, self.z_dim)
            self.dropout_proj = nn.Dropout(0.1)

    def get_edge_index(self, x):
        B, N, _ = x.shape

        knn_idx = get_knn_idx(x, x, self.k + 1)
        knn_idx = knn_idx[:, :, 1:]

        base = jt.arange(B) * N
        base = base.reshape(B, 1, 1)

        knn_idx = knn_idx + base

        dst = jt.arange(N).reshape(1, N, 1).broadcast((B, N, self.k))
        dst = dst + base

        src = knn_idx.reshape(-1)
        dst = dst.reshape(-1)

        edge_index = jt.stack([src, dst], dim=0)
        edge_index = remove_self_loops(edge_index)

        return edge_index

    def execute(self, x, disp_feat):
        B = x.shape[0]
        N = x.shape[1]

        if disp_feat is not None:
            disp_feat = nn.relu(self.linear_proj(disp_feat))
            disp_feat = self.dropout_proj(disp_feat)
            x = jt.concat([x, disp_feat], dim=-1)

        edge_index = self.get_edge_index(x)
        x_flat = x.reshape(B * N, -1)
        x1 = self.conv1(x_flat, edge_index)
        x1 = x1.reshape(B, N, -1)

        edge_index = self.get_edge_index(x1)
        x1_flat = x1.reshape(B * N, -1)
        x2 = self.conv2(x1_flat, edge_index)
        x2 = x2.reshape(B, N, -1)

        edge_index = self.get_edge_index(x2)
        x2_flat = x2.reshape(B * N, -1)
        x3 = self.conv3(x2_flat, edge_index)
        x3 = x3.reshape(B, N, -1)

        edge_index = self.get_edge_index(x3)
        x3_flat = x3.reshape(B * N, -1)
        x_combined = jt.concat([x1, x2, x3], dim=-1)
        x_combined_flat = x_combined.reshape(B * N, -1)
        x4 = self.conv4(x_combined_flat, edge_index)
        x4 = x4.reshape(B, N, -1)

        x = nn.relu(self.linear1(x4))
        x = nn.relu(self.linear2(x))
        x = x.mean(dim=1)
        x = jt.tanh(self.linear3(x))

        if self.z_dim > 0:
            combined = x_combined.transpose(2, 1)
            return x, combined
        else:
            return x, None
