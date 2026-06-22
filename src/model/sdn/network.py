"""Score-prediction network: feature fusion + gradient prediction + gradient
fusion (paper Sec. 4.1, Figure 2).

Given a *current* noisy point cloud ``x_t`` and a *reference* point cloud
``x_ref`` (the original noisy cloud ``x_T`` during inference, or the stage-1
cloud during training), the network predicts the per-point score vector
``s_theta(x_t | x_ref)`` -- the gradient pointing from each point toward the
underlying clean surface.

Pipeline:
    E(x_t), E(x_ref)                      feature extraction (see feature.py)
        |
    F = FeatureFusion(...)                Eq. 10  -- fuse current & reference
        |                                          features w.r.t. timestep t/T
    g_{v,i}, w_{v,i} = G(v - x_i, F_i)    Eq. 11  -- per-neighbour gradient +
        |                                          importance weight
    s(v) = sum_i g_{v,i} softmax(w)_i     Eq. 12  -- gradient fusion
"""

import math

import jittor as jt
from jittor import nn

from .feature import get_knn_idx


def gather_neighbors(feat, idx):
    """feat: (B, M, C), idx: (B, Q, k) -> (B, Q, k, C)."""
    B, M, C = feat.shape
    Q, k = idx.shape[1], idx.shape[2]
    out = []
    for b in range(B):
        out.append(feat[b][idx[b].reshape(-1)].reshape(Q, k, C))
    return jt.stack(out, dim=0)


def timestep_embedding(rel_t, dim):
    """Sinusoidal embedding of the relative timestep ``t / T`` in ``[0, 1]``.

    rel_t: (B,) scalar per cloud -> (B, dim)
    """
    half = dim // 2
    freqs = jt.exp(-math.log(10000.0) * jt.arange(half).float32() / max(half - 1, 1))
    args = rel_t.reshape(-1, 1).float32() * freqs.reshape(1, -1) * 1000.0
    emb = jt.concat([jt.cos(args), jt.sin(args)], dim=-1)
    if emb.shape[-1] < dim:                       # pad if dim is odd
        emb = jt.concat([emb, jt.zeros((emb.shape[0], dim - emb.shape[-1]))], dim=-1)
    return emb


class PositionalEncoding(nn.Module):
    """Fourier positional encoding of 3D coordinates."""

    def __init__(self, num_freqs=6):
        super().__init__()
        self.num_freqs = num_freqs
        self.out_dim = 3 * (2 * num_freqs + 1)
        # Plain python floats (not a jt.Var) so they are NOT registered as
        # trainable parameters -- these frequency bands are fixed constants.
        self.freq_bands = [float(2.0 ** i * math.pi) for i in range(num_freqs)]

    def execute(self, x):
        # x: (B, N, 3) -> (B, N, out_dim)
        feats = [x]
        for f in self.freq_bands:
            feats.append(jt.sin(x * f))
            feats.append(jt.cos(x * f))
        return jt.concat(feats, dim=-1)


def mlp3(in_dim, hidden, out_dim):
    """A 3-layer MLP (the paper fuses features with "corresponding 3-layer MLPs")."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class FeatureFusion(nn.Module):
    """Feature fusion module ``F`` (paper Eq. 10).

    ``F_t = MLP( E(x_t) * e_t + E(x_ref) * e_T )`` where the weight vectors
    ``e_t`` / ``e_T`` are produced from a shared embedding ``e`` of the
    positional encoding of ``x_t`` and the relative-timestep embedding.
    """

    def __init__(self, feat_dim, num_pos_freqs=6, time_dim=64):
        super().__init__()
        self.pe = PositionalEncoding(num_pos_freqs)
        self.time_dim = time_dim
        # e = MLP(positional_encoding(x_t) ++ time_embed(t/T))
        self.mlp_e = mlp3(self.pe.out_dim + time_dim, feat_dim, feat_dim)
        # weight vectors for current / reference features
        self.mlp_et = mlp3(feat_dim + feat_dim, feat_dim, feat_dim)
        self.mlp_eT = mlp3(feat_dim + feat_dim, feat_dim, feat_dim)
        # final fused feature
        self.mlp_f = mlp3(feat_dim, feat_dim, feat_dim)

    def execute(self, x_cur, rel_t, feat_cur, feat_ref):
        # x_cur: (B,N,3); rel_t: (B,) in [0,1]; feat_cur/feat_ref: (B,N,C)
        B, N, _ = x_cur.shape
        pe = self.pe(x_cur)                                   # (B,N,Cpe)
        te = timestep_embedding(rel_t, self.time_dim)         # (B,Ct)
        te = te.reshape(B, 1, self.time_dim).broadcast((B, N, self.time_dim))
        e = self.mlp_e(jt.concat([pe, te], dim=-1))           # (B,N,C)
        e_t = self.mlp_et(jt.concat([e, feat_cur], dim=-1))   # weight for current
        e_T = self.mlp_eT(jt.concat([e, feat_ref], dim=-1))   # weight for reference
        fused = self.mlp_f(feat_cur * e_t + feat_ref * e_T)   # Eq. 10
        return fused


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def execute(self, x):
        h = self.lin2(self.act(self.lin1(x)))
        return self.act(x + h)


class GradientPrediction(nn.Module):
    """Gradient prediction module ``G`` (paper Eq. 11).

    For a query point ``v`` and a correlated point ``x_i`` carrying fused
    feature ``F_i``, predict the gradient ``g_{v,i}`` of ``v`` and an importance
    weight ``w_{v,i}``. The query is first expressed relative to ``x_i``
    (``v - x_i``) for normalisation. Implemented as an MLP with 4 residual
    blocks.
    """

    def __init__(self, feat_dim, hidden=128, num_res=4):
        super().__init__()
        self.input = nn.Linear(3 + feat_dim, hidden)
        self.act = nn.ReLU()
        self.res = nn.ModuleList([ResidualBlock(hidden) for _ in range(num_res)])
        self.out_grad = nn.Linear(hidden, 3)        # g_{v,i}
        self.out_weight = nn.Linear(hidden, 1)      # w_{v,i} (importance)

    def execute(self, rel_coord, feat):
        # rel_coord: (..., 3), feat: (..., C)
        h = self.act(self.input(jt.concat([rel_coord, feat], dim=-1)))
        for blk in self.res:
            h = blk(h)
        return self.out_grad(h), self.out_weight(h)


class ScoreNetwork(nn.Module):
    """Full score predictor: E + FeatureFusion + GradientPrediction + fusion.

    ``predict_score`` evaluates ``s_theta(x_query | x_ref)`` for a set of query
    points, using the fused features of all points in ``x_cur`` as the support.
    """

    def __init__(self, feat_dim=128, k_graph=32, k_grad=32,
                 num_pos_freqs=6, time_dim=64, grad_hidden=128, num_res=4):
        super().__init__()
        from .feature import FeatureExtraction
        self.k_grad = k_grad
        self.feat_dim = feat_dim
        self.encoder = FeatureExtraction(k=k_graph, input_dim=3, embedding_dim=feat_dim)
        self.fusion = FeatureFusion(feat_dim, num_pos_freqs=num_pos_freqs, time_dim=time_dim)
        self.gradient = GradientPrediction(feat_dim, hidden=grad_hidden, num_res=num_res)

    def encode(self, x):
        return self.encoder(x)

    def fuse(self, x_cur, rel_t, feat_cur, feat_ref):
        return self.fusion(x_cur, rel_t, feat_cur, feat_ref)

    def predict_score(self, x_query, x_cur, fused_feat, k=None):
        """Gradient prediction + fusion (paper Eq. 11-13).

        x_query    : (B, Q, 3) points whose score we want
        x_cur      : (B, M, 3) support points (graph nodes)
        fused_feat : (B, M, C) fused feature of every support point
        returns    : (B, Q, 3) predicted score vectors
        """
        k = k or self.k_grad
        B, Q, _ = x_query.shape
        idx = get_knn_idx(x_query, x_cur, k)                   # (B,Q,k) neighbours in x_cur
        nbr_xyz = gather_neighbors(x_cur, idx)                 # (B,Q,k,3)
        nbr_feat = gather_neighbors(fused_feat, idx)           # (B,Q,k,C)
        rel = x_query.reshape(B, Q, 1, 3) - nbr_xyz            # v - x_i  (Eq. 11)

        g, w = self.gradient(rel, nbr_feat)                    # (B,Q,k,3), (B,Q,k,1)
        sw = nn.softmax(w, dim=2)                              # SoftMax over neighbours
        score = (g * sw).sum(dim=2)                            # weighted gradient fusion
        return score
