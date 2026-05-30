import math
import jittor as jt
from jittor import nn


# ---------- helper: KNN grouping (equivalent to pointops.queryandgroup) ----------

def knn_group(nsample, xyz, new_xyz, features, offset):
    """
    Equivalent to pointops.queryandgroup.
    Args:
        nsample: number of neighbors
        xyz: (total_N, 3) source points
        new_xyz: (total_N, 3) query points
        features: (total_N, d) source features
        offset: (B,) cumulative point counts, int32
    Returns:
        grouped_xyz: (total_N, nsample, 3)
        grouped_feat: (total_N, nsample, d)
    """
    total_N = xyz.shape[0]
    B = offset.shape[0]

    # reshape to batched (B, N_per, 3)
    N_per = offset[0].item()
    xyz_b = xyz.reshape(B, N_per, 3)
    new_xyz_b = new_xyz.reshape(B, N_per, 3)
    feat_b = features.reshape(B, N_per, -1)

    _, knn_idx = jt.misc.knn(new_xyz_b, xyz_b, nsample)  # (B, N_per, nsample)

    B_idx = jt.arange(B).unsqueeze(1).unsqueeze(2).broadcast((B, N_per, nsample))
    gathered_feat = feat_b[B_idx, knn_idx]  # (B, N_per, nsample, d)
    gathered_xyz = xyz_b[B_idx, knn_idx]    # (B, N_per, nsample, 3)

    gathered_feat = gathered_feat.reshape(total_N, nsample, -1)
    gathered_xyz = gathered_xyz.reshape(total_N, nsample, 3)
    return gathered_xyz, gathered_feat


# ---------- helper: FPS (equivalent to pointops.furthestsampling) ----------

def furthest_sampling(xyz, offset, n_offset):
    """
    Equivalent to pointops.furthestsampling.
    Args:
        xyz: (total_N, 3)
        offset: (B,) cumulative counts of input
        n_offset: (B,) cumulative counts of output
    Returns:
        indices: (total_M,) indices into xyz
    """
    B = offset.shape[0]
    off = [int(offset[i].item()) for i in range(B)]
    noff = [int(n_offset[i].item()) for i in range(B)]
    starts = [0] + off[:-1]
    sizes = [off[b] - starts[b] for b in range(B)]
    nsizes = [noff[b] - (0 if b == 0 else noff[b - 1]) for b in range(B)]

    # Fast path: all segments equal (always true inside the ASDN U-Net). FPS is
    # run batched with no per-iteration host sync, a large speedup over the
    # pure-Python loop while selecting exactly the same points.
    if len(set(sizes)) == 1 and len(set(nsizes)) == 1:
        N_per = sizes[0]
        npts = nsizes[0]
        pts = xyz.reshape(B, N_per, 3)
        dist = jt.ones((B, N_per)) * 1e10
        farthest = jt.zeros((B,), dtype='int32')  # start point index 0
        bar = jt.arange(B)
        selected = []
        for _ in range(npts):
            selected.append(farthest)
            centroid = pts[bar, farthest].reshape(B, 1, 3)
            d = ((pts - centroid) ** 2).sum(dim=2)
            dist = jt.minimum(dist, d)
            farthest = jt.argmax(dist, dim=1)[0].int32()
        idx = jt.stack(selected, dim=1)  # (B, npts) local indices
        base = jt.array(starts).int32().reshape(B, 1)
        return (idx + base).reshape(-1)

    # Fallback: unequal segments (kept for safety; not hit by the U-Net).
    indices = []
    for b in range(B):
        start, end = starts[b], off[b]
        npts = nsizes[b]
        pts = xyz[start:end]
        N_b = pts.shape[0]
        selected = []
        dist = jt.ones((N_b,)) * 1e10
        farthest = 0
        for _ in range(npts):
            selected.append(farthest)
            centroid = pts[farthest]
            d = ((pts - centroid) ** 2).sum(dim=1)
            dist = jt.minimum(dist, d)
            farthest, _ = jt.argmax(dist, dim=-1)
            farthest = farthest.item()
        idx = jt.array(selected).int32() + start
        indices.append(idx)
    return jt.concat(indices, dim=0)


# ---------- helper: interpolation (equivalent to pointops.interpolation) ----------

def feature_interpolation(xyz_dense, xyz_sparse, feat_sparse, offset_dense, offset_sparse, k=8):
    """
    Interpolate features from sparse points to dense points (inverse-distance weighted).
    Args:
        xyz_dense: (total_N_dense, 3)
        xyz_sparse: (total_N_sparse, 3)
        feat_sparse: (total_N_sparse, d)
        offset_dense: (B,) cumulative counts for dense
        offset_sparse: (B,) cumulative counts for sparse
        k: number of nearest neighbors
    Returns:
        interpolated: (total_N_dense, d)
    """
    B = offset_dense.shape[0]
    N_per_dense = offset_dense[0].item()
    N_per_sparse = offset_sparse[0].item()

    xyz_dense_b = xyz_dense.reshape(B, N_per_dense, 3)
    xyz_sparse_b = xyz_sparse.reshape(B, N_per_sparse, 3)
    feat_sparse_b = feat_sparse.reshape(B, N_per_sparse, -1)

    dist, knn_idx = jt.misc.knn(xyz_dense_b, xyz_sparse_b, k)  # (B, N_dense, k)
    dist = dist.maximum(1e-10)  # avoid division by zero
    weight = 1.0 / dist
    weight = weight / weight.sum(dim=-1, keepdims=True)  # normalize

    B_idx = jt.arange(B).unsqueeze(1).unsqueeze(2).broadcast((B, N_per_dense, k))
    gathered = feat_sparse_b[B_idx, knn_idx]  # (B, N_dense, k, d)
    interpolated = (gathered * weight.unsqueeze(-1)).sum(dim=2)  # (B, N_dense, d)
    return interpolated.reshape(-1, interpolated.shape[-1])


# ---------- helper: batched index select ----------

def batched_index_select(values, indices, dim=1):
    """
    Equivalent to batched_index_select in official code.
    values: (B, N, C)
    indices: (B, N, K)
    returns: (B, N, K, C)
    """
    B, N, C = values.shape
    _, _, K = indices.shape
    B_idx = jt.arange(B).unsqueeze(1).unsqueeze(2).broadcast((B, N, K))
    return values[B_idx, indices]


# ======================================================================
# Building Blocks
# ======================================================================


class StartBlock(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in + 3, d_out),
            nn.BatchNorm1d(d_out),
            nn.LeakyReLU(0.2),
        )

    def execute(self, p, x, o):
        x = self.mlp(p)
        return p, x, o


class RFE(nn.Module):
    """Residual Feature Extraction block."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.BiMLP = nn.Sequential(
            nn.Conv1d(10, 2 * d_out, 1),
            nn.BatchNorm1d(2 * d_out),
            nn.ReLU(),
            nn.Conv1d(2 * d_out, d_out, 1),
        )

        self.score_fn = nn.Sequential(
            nn.Linear(d_out * 2, d_out * 2, bias=False),
            nn.Softmax(dim=-2),
        )
        self.mlp_out = nn.Sequential(
            nn.Linear(d_out * 2, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(),
        )

    def execute(self, p, x):
        """
        p: (m, 16, 3)  neighborhood coordinates
        x: (m, 16, d_in)  neighborhood features
        returns: (m, d_out)
        """
        m, nsample = p.shape[0], p.shape[1]

        extended_coords = p[:, 0, :].unsqueeze(1).broadcast((m, nsample, 3))
        neighbors = p
        diff = extended_coords - neighbors
        dist = jt.sqrt((diff ** 2).sum(dim=2, keepdims=True))
        concat = jt.concat([extended_coords, neighbors, diff, dist], dim=-1)  # (m, 16, 10)

        p_c = self.BiMLP(concat.permute(0, 2, 1)).permute(0, 2, 1)

        p_x = jt.concat([p_c, x], dim=-1)  # (m, 16, d_out*2)
        scores = self.score_fn(p_x)
        features = (scores * p_x).sum(dim=1, keepdims=True)  # (m, 1, d_out*2)
        features = self.mlp_out(features.squeeze(dim=1))
        return features


class MRE(nn.Module):
    """Multi-scale Residual Extraction block."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.mlp0 = nn.Sequential(
            nn.Linear(d_in, d_out // 2),
            nn.BatchNorm1d(d_out // 2),
            nn.ReLU(),
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(d_out, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(),
        )
        self.mlp01 = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(),
        )
        self.Rfe_1 = RFE(d_out // 2, d_out // 2)
        self.Rfe_2 = RFE(d_out // 2, d_out // 2)

    def execute(self, p, x, o):
        x_start = x
        x = self.mlp0(x)

        grouped_xyz, grouped_feat = knn_group(16, p, p, x, o)
        xr = jt.concat([grouped_xyz, grouped_feat], dim=-1)  # (N, 16, 3 + d_out//2)
        x = self.Rfe_1(xr[:, :, :3], xr[:, :, 3:])

        x_middle = x
        grouped_xyz2, grouped_feat2 = knn_group(16, p, p, x, o)
        xr2 = jt.concat([grouped_xyz2, grouped_feat2], dim=-1)
        x = self.Rfe_2(xr2[:, :, :3], xr2[:, :, 3:])

        x = jt.concat([x_middle, x], dim=1)
        x = self.mlp01(x_start) + self.mlp1(x)
        return x


class Downsampling(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.nsample = nsample
        self.stride = stride
        self.mre = MRE(d_in, d_out)

    def execute(self, p, x, o):
        x = self.mre(p, x, o)

        # FPS downsampling
        count = o[0].item() * self.stride // (self.stride + 1)
        n_o = jt.array([count * (i + 1) for i in range(o.shape[0])]).int32()
        idx = furthest_sampling(p, o, n_o)

        n_p = p[idx, :]
        n_x = x[idx, :]

        return n_p, n_x, n_o


class CrossAttentionPointTransformerLayer(nn.Module):
    def __init__(self, dim, attn_mlp_hidden_mult=4, num_neighbors=None):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.attn_bimlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.BatchNorm1d(dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )

    def execute(self, x_e, x_r, x_d, pos):
        """
        x_e: (B, N, C1) dense features as Q
        x_r: (B, N, C2) interpolated features as K
        x_d: (B, N, C2) interpolated features as V
        pos: (B, N, 3) dense positions
        """
        n = x_e.shape[1]

        q = self.to_q(x_e)          # (B, N, C)
        k = self.to_k(x_r)          # (B, N, C)
        v = self.to_v(x_d)          # (B, N, C)

        if self.num_neighbors is not None and self.num_neighbors < n:
            rel_pos = pos[:, :, None, :] - pos[:, None, :, :]  # (B, N, N, 3)
            rel_dist = rel_pos.norm(dim=-1)                     # (B, N, N)
            _, indices = jt.topk(rel_dist, k=self.num_neighbors, dim=-1, largest=False)

            v = batched_index_select(v, indices, dim=1)
            k = batched_index_select(k, indices, dim=1)
            qk_rel = q[:, :, None, :] - k                       # (B, N, K, C)
            x_e_sel = batched_index_select(x_e, indices, dim=1)

        v = v + x_e_sel

        B, N, neigh_num, C = qk_rel.shape
        sim = self.attn_bimlp(
            (qk_rel + x_e_sel).reshape(B * N * neigh_num, C)
        ).reshape(B, N, neigh_num, C)

        attn = sim.softmax(dim=-2)
        agg = (attn * v).sum(dim=2)
        return agg


class Upsampling(nn.Module):
    def __init__(self, d_in_sparse_fusion, d_out, nsample):
        super().__init__()
        d_in_sparse, d_in_dense = d_in_sparse_fusion
        self.nsample = nsample
        self.d_out = d_out

        self.CrossPT_func = CrossAttentionPointTransformerLayer(
            dim=d_in_sparse, attn_mlp_hidden_mult=1, num_neighbors=16
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_in_sparse + d_in_dense, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(),
        )

        self.dense_mlp = nn.Sequential(
            nn.Linear(d_in_dense, d_in_sparse),
            nn.BatchNorm1d(d_in_sparse),
            nn.ReLU(),
        )

    def execute(self, p1, x1, o1, p2, x2, o2, batch_size):
        """
        p1, x1, o1: dense (higher resolution)
        p2, x2, o2: sparse (lower resolution)
        """
        num_points = p1.shape[0] // batch_size

        x1_dense = self.dense_mlp(x1)
        x2_interpolated = feature_interpolation(p1, p2, x2, o1, o2, k=8)

        x1_enhance = self.CrossPT_func(
            x1_dense.reshape(batch_size, num_points, -1),
            x2_interpolated.reshape(batch_size, num_points, -1),
            x2_interpolated.reshape(batch_size, num_points, -1),
            p1.reshape(batch_size, num_points, 3),
        ).reshape(batch_size * num_points, -1)

        x = self.mlp(jt.concat([x1_enhance, x1], dim=-1))
        return p1, x, o1


def block_decider(name):
    if name == 'startblock':
        return StartBlock
    if name == 'upsample':
        return Upsampling
    if name == 'downsample':
        return Downsampling
    raise ValueError(f"Unknown block: {name}")
