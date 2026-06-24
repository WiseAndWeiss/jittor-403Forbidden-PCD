"""PDFlowModule -- denoising via an invertible normalizing flow with latent
noise disentanglement.

Jittor reproduction of the *core idea* of PD-LTS / PD-Flow (Mao et al., CVPR
2024 / 2022): augment each point with learned features, map the cloud through an
invertible flow conditioned on graph-convolution features, **zero out a noise
subspace in the latent code** (fixed binary mask), and invert the flow to
recover the clean coordinates. Trained directly with Chamfer distance, which
aligns with the competition metric.

The flow here uses closed-form ActNorm + affine-coupling blocks (flow.py) instead
of PD-LTS's implicit monotone residual flow -- robust and Jittor-friendly while
keeping the disentanglement principle.
"""

from math import ceil
from typing import Dict, List

import jittor as jt
import numpy as np

from .feature import noiseEdgeConv, PreConv, EdgeConv, FeatMergeUnit, get_knn_idx
from .flow import FlowBlock
from ..spec import ModelSpec
from ...data.asset import Asset


def chamfer_distance(a, b):
    """Symmetric Chamfer distance between (B,N,3) and (B,M,3)."""
    diff = a.unsqueeze(2) - b.unsqueeze(1)          # (B,N,M,3)
    dist = (diff ** 2).sum(-1)                      # (B,N,M)
    return jt.min(dist, dim=2).mean() + jt.min(dist, dim=1).mean()


def sinkhorn_emd(x, y, eps=0.01, n_iter=50):
    """Differentiable entropic-OT (Sinkhorn) approximation of Earth-Mover's
    distance between two equal-size clouds (B,N,3). Unlike Chamfer (which allows
    many-to-one matches and tolerates clustering/gaps), EMD enforces a balanced
    1-to-1 transport -> rewards a uniform point distribution, directly targeting
    the CD-coverage bottleneck. This is the loss PD-LTS / PD-Flow use."""
    B, N, _ = x.shape
    C = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)        # (B,N,N) cost (grad via x)
    # Solve for the optimal transport plan WITHOUT grad (envelope theorem): backprop
    # through 50 unrolled Sinkhorn steps is huge; instead freeze the plan and let the
    # gradient flow only through the cost C -> a stable soft-EMD matching loss.
    with jt.no_grad():
        K = jt.exp(-C / eps)                                   # Gibbs kernel
        a = jt.ones((B, N, 1)) / N                             # uniform marginals
        u = jt.ones((B, N, 1))
        for _ in range(n_iter):
            v = a / (jt.matmul(K.transpose(1, 2), u) + 1e-8)
            u = a / (jt.matmul(K, v) + 1e-8)
        T = u * K * v.transpose(1, 2)                          # transport plan (B,N,N)
    return (T * C).sum(dims=[1, 2]).mean()


class PDFlowModule(ModelSpec):

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config

        self.pc_channel = 3
        self.aug_channel = cfg.get('aug_channel', 33)           # -> total channel even
        self.channel = self.pc_channel + self.aug_channel
        assert self.channel % 2 == 0, "pc+aug channel must be even"
        self.cut_channel = cfg.get('cut_channel', 6)            # noise subspace to zero
        self.k = cfg.get('num_neighbors', 32)
        self.n_injector = cfg.get('n_injector', 7)
        self.nflow = cfg.get('nflow_module', 8)
        feat_dim = cfg.get('feat_dim', 32)
        hidden = cfg.get('flow_hidden', 64)

        # ---- loss: Chamfer + Sinkhorn-EMD (distribution) ----
        self.cd_weight = cfg.get('cd_weight', 1.0)
        self.sinkhorn_weight = cfg.get('sinkhorn_weight', 0.0)
        self.sinkhorn_eps = cfg.get('sinkhorn_eps', 0.01)
        self.sinkhorn_iters = cfg.get('sinkhorn_iters', 50)

        # ---- augmentation + graph-conv feature injectors ----
        self.noise_params = noiseEdgeConv(self.pc_channel, hidden_channel=feat_dim,
                                          out_channel=self.aug_channel)
        self.pre = PreConv(self.pc_channel, feat_dim)
        self.feat_conv = jt.nn.ModuleList()
        self.merge = jt.nn.ModuleList()
        for _ in range(self.n_injector):
            self.feat_conv.append(EdgeConv(feat_dim, hidden_channel=64,
                                           out_channel=feat_dim, concat=False))
            self.merge.append(FeatMergeUnit(feat_dim, hidden_channel=64,
                                            out_channel=self.channel))

        # ---- invertible flow ----
        self.blocks = jt.nn.ModuleList([
            FlowBlock(self.channel, hidden=hidden, swap=(i % 2 == 1))
            for i in range(self.nflow)
        ])

        # ---- inference ----
        self.predict_patch_size = cfg.get('predict_patch_size', 1024)
        self.predict_seed_k = cfg.get('predict_seed_k', 3)
        self.predict_iters = cfg.get('predict_iters', 1)
        self.predict_patch_step = cfg.get('predict_patch_step', 16)

    # ------------------------------------------------------------------ #
    def feat_extract(self, xyz):
        knn_idx = get_knn_idx(xyz, self.k)
        f = self.pre(xyz, knn_idx)
        injectors = []
        for i in range(self.n_injector):
            f = self.feat_conv[i](f, knn_idx)
            injectors.append(self.merge[i](f))
        return injectors

    def augment(self, xyz):
        knn_idx = get_knn_idx(xyz, self.k)
        return self.noise_params(xyz, knn_idx)                  # (B,N,aug)

    def f(self, x, inj):
        for i in range(self.nflow):
            if i < self.n_injector:
                x = x + inj[i]
            x = self.blocks[i](x)
        return x

    def g(self, z, inj):
        for i in reversed(range(self.nflow)):
            z = self.blocks[i].inverse(z)
            if i < self.n_injector:
                z = z - inj[i]
        return z

    def denoise(self, noisy):
        """noisy: (B,N,3) -> clean (B,N,3)."""
        aug = self.augment(noisy)
        x = jt.concat([noisy, aug], dim=-1)                     # (B,N,channel)
        inj = self.feat_extract(noisy)
        z = self.f(x, inj)
        # fixed binary mask: zero the noise subspace (latent disentanglement)
        z[..., -self.cut_channel:] = 0.0
        x_rec = self.g(z, inj)
        return x_rec[..., :self.pc_channel]

    # ------------------------------------------------------------------ #
    def get_supervised_loss(self, pc_noisy, pc_clean):
        denoised = self.denoise(pc_noisy)
        cd = chamfer_distance(denoised, pc_clean)
        if self.sinkhorn_weight > 0:
            emd = sinkhorn_emd(denoised, pc_clean, eps=self.sinkhorn_eps,
                               n_iter=self.sinkhorn_iters)
        else:
            emd = jt.array(0.0)
        loss = self.cd_weight * cd + self.sinkhorn_weight * emd
        return loss, cd, emd

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        loss, cd, emd = self.get_supervised_loss(pc_noisy, pc_clean)
        return {"loss": loss, "cd_loss": cd, "emd_loss": emd}

    def execute(self, **kwargs) -> Dict:    # type: ignore
        return self.training_step(**kwargs)

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({
                    "pc_noisy": b.meta['pc_noisy'],
                    "pc_clean": b.meta['pc_clean'],
                })
            else:
                d = {"pc_noisy": b.sampled_vertices_noisy}
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                if b.meta is not None:
                    d["norm_center"] = np.array(b.meta.get('norm_center'))
                    d["norm_scale"] = np.array(b.meta.get('norm_scale'))
                res.append(d)
        return res

    # ------------------------------------------------------------------ #
    # patch-based inference
    # ------------------------------------------------------------------ #
    @jt.no_grad()
    def _denoise_full(self, pcl_noisy):
        """Patch the (N,3) cloud, flow-denoise each patch, stitch by nearest seed."""
        N = pcl_noisy.shape[0]
        M = self.predict_patch_size
        num_patches = max(int(self.predict_seed_k * N / M), 1)
        seed_idx = farthest_point_sampling(pcl_noisy, num_patches)
        seed = pcl_noisy[seed_idx].unsqueeze(0)                 # (1,P,3)
        _, pidx = jt.misc.knn(seed, pcl_noisy.unsqueeze(0), M)  # (1,P,M)
        pidx = pidx[0]                                          # (P,M)

        patches = jt.stack([pcl_noisy[pidx[p]] for p in range(num_patches)], dim=0)
        seed_exp = seed.squeeze(0).unsqueeze(1)                 # (P,1,3)
        patches_c = patches - seed_exp                          # center

        # nearest-seed assignment for stitching
        d2 = ((patches - seed_exp) ** 2).sum(-1)               # (P,M)
        all_d = jt.ones((num_patches, N)) * 1e10
        for p in range(num_patches):
            all_d[p][pidx[p]] = d2[p]
        best = jt.argmin(all_d, dim=0)[0]                       # (N,)

        out = pcl_noisy.clone()
        step = self.predict_patch_step
        den_all = []
        i = 0
        while i < num_patches:
            den_all.append(self.denoise(patches_c[i:i + step]))
            i += step
        den_all = jt.concat(den_all, dim=0) + seed_exp          # (P,M,3)
        for p in range(num_patches):
            idxs = pidx[p]
            sel = (best[idxs] == p)
            if sel.sum().item() > 0:
                out[idxs[sel]] = den_all[p][sel]
        return out

    @jt.no_grad()
    def adaptive_denoise(self, pcl_noisy):
        x = pcl_noisy
        for _ in range(self.predict_iters):
            x = self._denoise_full(x)
        return x

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            den = self.adaptive_denoise(pc_noisy).detach().numpy()
            if 'norm_center' in batch and 'norm_scale' in batch:
                c = batch['norm_center'][i]; s = batch['norm_scale'][i]
                if isinstance(c, jt.Var): c = c.numpy()
                if isinstance(s, jt.Var): s = s.numpy()
                den = den * s + c
            res.append({"pc_denoised": den})
        return res


def farthest_point_sampling(pcl, num_pnts):
    N = pcl.shape[0]
    dist = jt.ones((N,)) * 1e10
    selected = []
    farthest = 0
    for _ in range(num_pnts):
        selected.append(farthest)
        d = ((pcl - pcl[farthest]) ** 2).sum(dim=1)
        dist = jt.minimum(dist, d)
        farthest, _ = jt.argmax(dist, dim=-1)
        farthest = int(farthest.item())
    return jt.array(selected).int32()
