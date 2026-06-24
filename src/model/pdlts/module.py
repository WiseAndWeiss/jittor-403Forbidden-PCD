"""PDLTSModule -- faithful Jittor reproduction of PD-LTS (CVPR 2024) light model.

Architecture matches yanbiao1/PD-LTS models/model_light/deflow.py::DenoiseFlow with
the FBM (fixed binary mask) disentanglement:
  - noiseEdgeConv augmentation + EdgeConv/FeatMergeUnit graph-conv feature injectors
    (growing-concat channel schedule, identical to PD-LTS),
  - 8 FlowAssembly modules, each [MonotoneBlock, ActNorm, MonotoneBlock(preact),
    ActNorm] -- a spectrally-normalized implicit *monotone residual* flow (the real
    PD-LTS engine, not affine coupling),
  - FBM: zero the last `cut_channel` latent channels, invert the flow.

The log-det-Jacobian is unused for FBM denoising (PD-LTS hard-codes logp=0), so it is
dropped. Training loss is EMD (hard-assignment) as in the paper; a CD option is kept
for validating the flow port. Inference is patch-based, centered per seed.
"""

from typing import Dict, List

import jittor as jt
import numpy as np

from ..pdf.feature import noiseEdgeConv, PreConv, EdgeConv, FeatMergeUnit, get_knn_idx, knn_gather
from ..pdf.module import chamfer_distance, farthest_point_sampling
from .monotone import FlowAssembly
from ..spec import ModelSpec
from ...data.asset import Asset


def hard_emd_loss(pred, clean, eps=0.003, n_iter=60):
    """EMD-style loss via a near-permutation assignment. A low-eps Sinkhorn plan is
    solved WITHOUT grad, the hard row-assignment is read off (argmax), and the loss is
    the matched squared distance. Unlike a soft Sinkhorn cost this does NOT pull points
    to blurred barycenters -- each pred point gets one concrete target (bijection-ish),
    matching PD-LTS's auction EMD behavior while staying cheap/differentiable."""
    B, N, _ = pred.shape
    with jt.no_grad():
        C = ((pred.unsqueeze(2) - clean.unsqueeze(1)) ** 2).sum(-1)   # (B,N,M)
        K = jt.exp(-C / eps)
        a = jt.ones((B, N, 1)) / N
        u = jt.ones((B, N, 1))
        for _ in range(n_iter):
            v = a / (jt.matmul(K.transpose(0, 2, 1), u) + 1e-8)
            u = a / (jt.matmul(K, v) + 1e-8)
        T = u * K * v.transpose(0, 2, 1)                              # (B,N,M) plan
        assign = jt.argmax(T, dim=2)[0]                              # (B,N) target idx
    matched = knn_gather(clean, assign.unsqueeze(-1))[:, :, 0]        # (B,N,3)
    return ((pred - matched) ** 2).sum(-1).mean()


class PDLTSModule(ModelSpec):

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config

        self.pc_channel = 3
        self.aug_channel = cfg.get('aug_channel', 32)
        self.channel = self.pc_channel + self.aug_channel
        self.cut_channel = cfg.get('cut_channel', 6)
        self.k = cfg.get('num_neighbors', 32)
        self.n_injector = cfg.get('n_injector', 7)
        self.nflow = cfg.get('nflow_module', 8)
        idim = cfg.get('idim', 64)
        nhidden = cfg.get('nhidden', 2)
        coeff = cfg.get('coeff', 0.9)
        n_fixed_iter = cfg.get('n_fixed_iter', 30)

        # ---- loss ----
        self.loss_type = cfg.get('loss_type', 'emd')          # 'emd' | 'cd'
        self.emd_eps = cfg.get('emd_eps', 0.003)
        self.emd_iters = cfg.get('emd_iters', 60)

        # ---- augmentation + graph-conv feature injectors (PD-LTS schedule) ----
        self.noise_params = noiseEdgeConv(self.pc_channel, hidden_channel=32,
                                          out_channel=self.aug_channel)
        self.pre = PreConv(self.pc_channel, 16)
        self.feat_conv = jt.nn.ModuleList()
        self.merge = jt.nn.ModuleList()
        for i in range(self.n_injector):
            in_e = 16 + 32 * i                                # growing concat dim
            self.feat_conv.append(EdgeConv(in_e, hidden_channel=64,
                                           out_channel=32, concat=True))
            self.merge.append(FeatMergeUnit(in_e + 32, hidden_channel=64,
                                            out_channel=self.channel))

        # ---- invertible monotone residual flow ----
        self.blocks = jt.nn.ModuleList([
            FlowAssembly(self.channel, idim=idim, nhidden=nhidden, coeff=coeff,
                         n_fixed_iter=n_fixed_iter)
            for _ in range(self.nflow)
        ])

        # ---- inference ----
        self.predict_patch_size = cfg.get('predict_patch_size', 1024)
        self.predict_seed_k = cfg.get('predict_seed_k', 3)
        self.predict_iters = cfg.get('predict_iters', 1)
        self.predict_patch_step = cfg.get('predict_patch_step', 8)

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
        return self.noise_params(xyz, knn_idx)

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
        aug = self.augment(noisy)
        x = jt.concat([noisy, aug], dim=-1)
        inj = self.feat_extract(noisy)
        z = self.f(x, inj)
        z[..., -self.cut_channel:] = 0.0                     # FBM disentangle
        x_rec = self.g(z, inj)
        return x_rec[..., :self.pc_channel]

    # ------------------------------------------------------------------ #
    def get_supervised_loss(self, pc_noisy, pc_clean):
        denoised = self.denoise(pc_noisy)
        if self.loss_type == 'cd':
            loss = chamfer_distance(denoised, pc_clean)
        else:
            loss = hard_emd_loss(denoised, pc_clean, eps=self.emd_eps, n_iter=self.emd_iters)
        return loss

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(pc_noisy, pc_clean)
        return {"loss": loss}

    def execute(self, **kwargs) -> Dict:        # type: ignore
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
    @jt.no_grad()
    def _denoise_full(self, pcl_noisy):
        N = pcl_noisy.shape[0]
        M = self.predict_patch_size
        num_patches = max(int(self.predict_seed_k * N / M), 1)
        seed_idx = farthest_point_sampling(pcl_noisy, num_patches)
        seed = pcl_noisy[seed_idx].unsqueeze(0)
        _, pidx = jt.misc.knn(seed, pcl_noisy.unsqueeze(0), M)
        pidx = pidx[0]

        patches = jt.stack([pcl_noisy[pidx[p]] for p in range(num_patches)], dim=0)
        seed_exp = seed.squeeze(0).unsqueeze(1)
        patches_c = patches - seed_exp                       # center per seed

        d2 = ((patches - seed_exp) ** 2).sum(-1)
        all_d = jt.ones((num_patches, N)) * 1e10
        for p in range(num_patches):
            all_d[p][pidx[p]] = d2[p]
        best = jt.argmin(all_d, dim=0)[0]

        out = pcl_noisy.clone()
        step = self.predict_patch_step
        den_all = []
        i = 0
        while i < num_patches:
            den_all.append(self.denoise(patches_c[i:i + step]))
            i += step
        den_all = jt.concat(den_all, dim=0) + seed_exp
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
        self.init_as_trained_state()        # use loaded ActNorm params, don't re-init on eval data
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

    def init_as_trained_state(self):
        """ActNorm must be marked initialized before eval with a loaded checkpoint."""
        for blk in self.blocks:
            for m in blk.chain:
                if hasattr(m, 'inited'):
                    m.inited = True
