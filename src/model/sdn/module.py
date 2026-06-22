"""SDNModule -- Adaptive and Iterative Score-Based Point Cloud Denoiser.

Faithful Jittor reproduction of
"Adaptive and Iterative Point Cloud Denoising with Score-Based Diffusion Model"
(Wang, Li, Xin, Tu; 2025).

The three paper innovations live here:
  * Score-based diffusion with **scaling elimination** (Sec 3.1) -> diffusion.py
  * **Two-stage sampling** training (Algorithm 1)                -> training_step
  * **Adaptive & iterative** denoising schedule (Algorithm 2)    -> predict_step
"""

from math import ceil
from typing import Dict, List

import jittor as jt
import numpy as np

from .diffusion import DiffusionSchedule
from .network import ScoreNetwork
from ..spec import ModelSpec
from ...data.asset import Asset


# --------------------------------------------------------------------------- #
# small geometry helpers
# --------------------------------------------------------------------------- #
def nn_displacement(query, ref):
    """Ground-truth score approximation S(x) = NN(x, ref) - x  (paper Eq. 14).

    query: (B, Q, 3) noisy points ; ref: (B, M, 3) clean points
    returns (B, Q, 3): displacement from each query point to its nearest clean
    neighbour (points toward the underlying surface).
    """
    _, idx = jt.misc.knn(query, ref, 1)              # (B, Q, 1)
    B, Q, _ = query.shape
    nn_pts = []
    for b in range(B):
        nn_pts.append(ref[b][idx[b].reshape(-1)].reshape(Q, 3))
    nn_pts = jt.stack(nn_pts, dim=0)
    return nn_pts - query


def farthest_point_sampling(pcl, num_pnts):
    """Vectorised-ish FPS for a single (N,3) cloud -> indices (num_pnts,)."""
    N = pcl.shape[0]
    dist = jt.ones((N,)) * 1e10
    selected = []
    farthest = 0
    for _ in range(num_pnts):
        selected.append(farthest)
        centroid = pcl[farthest]
        d = ((pcl - centroid) ** 2).sum(dim=1)
        dist = jt.minimum(dist, d)
        farthest, _ = jt.argmax(dist, dim=-1)
        farthest = int(farthest.item())
    return jt.array(selected).int32()


class SDNModule(ModelSpec):

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config

        # ---- diffusion schedule (scaling-elimination form) ----
        self.T = cfg.get('num_diffusion_steps', 1000)
        self.beta_T = cfg.get('beta_T', 2e-6)
        self.schedule = DiffusionSchedule(num_steps=self.T, beta_T=self.beta_T)

        # ---- training hyper-parameters (Algorithm 1) ----
        self.patch_size = cfg.get('patch_size', 1000)       # points per training patch
        self.kp_mask = cfg.get('kp_mask', 256)              # K_p highlighted points
        self.lam = cfg.get('lambda_weight', 0.99)           # loss weight lambda (Eq. 9)
        self.t_min = cfg.get('t_min', 1)                    # min sampled timestep
        # The interpolated weight (1-lambda)/sigma_bar + lambda diverges as
        # sigma_bar -> 0 (the clean end, reachable when stage-2 samples
        # Delta = t -> s = 0). Floor sigma_bar in the weight at the smallest
        # *trained* noise scale so the weight stays bounded and the loss does not
        # spike to ~1e7 on those samples. Faithful to the paper for all real
        # timesteps; only regularises the t->0 singularity.
        floor = cfg.get('weight_sigma_floor', None)
        self.weight_sigma_floor = float(floor) if floor is not None \
            else float(self.schedule.sigma_of(self.t_min))

        # ---- network ----
        self.feat_dim = cfg.get('feat_dim', 128)
        self.k_graph = cfg.get('k_graph', 32)               # DGCNN graph k
        self.k_grad = cfg.get('k_grad', 32)                 # gradient-fusion neighbours
        self.net = ScoreNetwork(
            feat_dim=self.feat_dim,
            k_graph=self.k_graph,
            k_grad=self.k_grad,
            num_pos_freqs=cfg.get('num_pos_freqs', 6),
            time_dim=cfg.get('time_dim', 64),
            grad_hidden=cfg.get('grad_hidden', 128),
            num_res=cfg.get('num_res_blocks', 4),
        )

        # ---- inference hyper-parameters (Algorithm 2) ----
        self.L = cfg.get('num_iters', 5)                    # adaptive schedule length
        self.predict_patch_size = cfg.get('predict_patch_size', 1000)
        self.predict_seed_k = cfg.get('predict_seed_k', 3)  # patch overlap factor
        self.predict_patch_step = cfg.get('predict_patch_step', 16)  # patches / batch
        # Var(||score||) underestimates the per-coordinate noise variance sigma_bar^2
        # (for 3D Gaussian noise Var(||.||) ~ 0.45 sigma^2), so the estimated tau is
        # too small and denoising too weak. Scale the noise estimate to denoise
        # harder; sweep this on the local eval. 1.0 = paper-literal.
        self.predict_noise_scale = cfg.get('predict_noise_scale', 1.0)
        # Optional output post-processing (outlier pull-in + light normal-direction
        # smoothing). Cheap, training-independent; can improve CD/P2S. Off by default.
        self.predict_post_smooth = cfg.get('predict_post_smooth', False)
        self.pca_smooth_alpha = cfg.get('pca_smooth_alpha', 0.0)

    # ------------------------------------------------------------------ #
    # data plumbing
    # ------------------------------------------------------------------ #
    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({
                    "patch_clean": b.meta['patch_clean'],   # (P, M, 3) clean patches
                    "mask": b.meta['mask'],                 # (P, M) highlighted K_p mask
                })
            else:
                d = {"pc_noisy": b.sampled_vertices_noisy}  # (N, 3)
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                if b.meta is not None:
                    d["norm_center"] = np.array(b.meta.get('norm_center'))
                    d["norm_scale"] = np.array(b.meta.get('norm_scale'))
                res.append(d)
        return res

    # ------------------------------------------------------------------ #
    # Training -- two-stage sampling (Algorithm 1)
    # ------------------------------------------------------------------ #
    def get_supervised_loss(self, x0, mask):
        """x0: (G, M, 3) clean patches ; mask: (G, M) in {0,1} (first K_p = 1)."""
        G, M, _ = x0.shape
        Kp = self.kp_mask

        # ---------------- Stage 1: sample x_t and predict s(x_t | x_t) -----
        t = np.random.randint(self.t_min, self.T + 1, size=G)     # per-patch timestep
        sigma_t = self.schedule.sigma_of(t)                       # (G,)
        sigma_t_v = jt.array(sigma_t.astype(np.float32)).reshape(G, 1, 1)
        eps = jt.randn(x0.shape) * sigma_t_v
        x_t = x0 + eps                                            # x_t = x_0 + sigma_t z

        feat_t = self.net.encode(x_t)
        rel_t1 = jt.ones((G,))                                    # t / t = 1 (stage 1)
        F_t = self.net.fuse(x_t, rel_t1, feat_t, feat_t)         # ref == current

        # GT scores everywhere (needed for the non-masked region of the update)
        S_t = nn_displacement(x_t, x0)                            # (G, M, 3)
        # predicted score only at the K_p highlighted (masked) points
        x_q1 = x_t[:, :Kp, :]
        score_t = self.net.predict_score(x_q1, x_t, F_t)         # (G, Kp, 3)

        sigma_t_w = np.maximum(sigma_t, self.weight_sigma_floor)
        w_t = jt.array(((1.0 - self.lam) / sigma_t_w + self.lam).astype(np.float32)).reshape(G, 1, 1)
        loss_t = ((w_t * (score_t - S_t[:, :Kp, :])) ** 2).sum(dim=-1).mean()

        # ---------------- Stage 2: sample x_{t-d} and predict s(x_{t-d}|x_t) ---
        delta = np.array([np.random.randint(1, int(ti) + 1) for ti in t])   # Delta ~ U(1..t)
        s_idx = (t - delta).astype(np.int64)                     # t - Delta >= 0
        coef = self.schedule.update_coef(t, s_idx)               # Eq. 7 multiplier
        coef_v = jt.array(coef.astype(np.float32)).reshape(G, 1, 1)

        # blended score: predicted on masked region, GT elsewhere (Algorithm 1, line 10)
        blended = S_t.clone()
        blended[:, :Kp, :] = score_t
        x_s = x_t + coef_v * blended

        feat_s = self.net.encode(x_s)
        rel_t2 = jt.array((s_idx / np.maximum(t, 1)).astype(np.float32))     # (t-Delta)/t
        F_s = self.net.fuse(x_s, rel_t2, feat_s, feat_t)         # ref == stage-1 cloud x_t
        S_s = nn_displacement(x_s, x0)
        x_q2 = x_s[:, :Kp, :]
        score_s = self.net.predict_score(x_q2, x_s, F_s)

        sigma_s = self.schedule.sigma_of(s_idx)
        sigma_s = np.maximum(sigma_s, self.weight_sigma_floor)   # floor weight singularity at s->0
        w_s = jt.array(((1.0 - self.lam) / sigma_s + self.lam).astype(np.float32)).reshape(G, 1, 1)
        loss_s = ((w_s * (score_s - S_s[:, :Kp, :])) ** 2).sum(dim=-1).mean()

        loss = loss_t + loss_s
        return loss, loss_t, loss_s

    def training_step(self, batch: Dict) -> Dict:
        M = batch['patch_clean'].shape[-2]
        x0 = batch['patch_clean'].reshape(-1, M, 3)
        mask = batch['mask'].reshape(-1, M)
        loss, loss_t, loss_s = self.get_supervised_loss(x0, mask)
        return {"loss": loss, "score_loss_t": loss_t, "score_loss_s": loss_s}

    def execute(self, **kwargs) -> Dict:    # type: ignore
        return self.training_step(**kwargs)

    # ------------------------------------------------------------------ #
    # Inference -- adaptive & iterative denoising (Algorithm 2)
    # ------------------------------------------------------------------ #
    @jt.no_grad()
    def _build_patches(self, pcl_noisy):
        """FPS overlapping patches (mirrors the score-denoise patch scheme)."""
        N = pcl_noisy.shape[0]
        patch_size = self.predict_patch_size
        num_patches = int(self.predict_seed_k * N / patch_size)
        num_patches = max(num_patches, 1)
        pcl = pcl_noisy.unsqueeze(0)                             # (1, N, 3)

        seed_idx = farthest_point_sampling(pcl_noisy, num_patches)   # (P,)
        seed_pnts = pcl_noisy[seed_idx].unsqueeze(0)            # (1, P, 3)
        _, point_idxs = jt.misc.knn(seed_pnts, pcl, patch_size)  # (1, P, M)
        point_idxs = point_idxs[0]                              # (P, M)

        patches = []
        for p in range(num_patches):
            patches.append(pcl_noisy[point_idxs[p]])
        patches = jt.stack(patches, dim=0)                     # (P, M, 3)
        seed_expand = seed_pnts.squeeze(0).unsqueeze(1)        # (P, 1, 3)
        patches_centered = patches - seed_expand               # center each patch

        # nearest-patch assignment for stitching (closest to a patch center wins)
        dist_to_seed = ((patches - seed_expand) ** 2).sum(-1)  # (P, M)
        all_dists = jt.ones((num_patches, N)) * 1e10
        for p in range(num_patches):
            all_dists[p][point_idxs[p]] = dist_to_seed[p]
        best_patch = jt.argmin(all_dists, dim=0)[0]            # (N,) patch id per point
        return patches_centered, seed_expand, point_idxs, best_patch, num_patches

    @jt.no_grad()
    def _estimate_noise_var(self, patches_centered):
        """Primary noise-variance guess sigma^2 = Var(||s_theta(X|X)||) (Eq. 15)."""
        step = self.predict_patch_step
        norms = []
        i = 0
        P = patches_centered.shape[0]
        while i < P:
            x = patches_centered[i:i + step]                   # (b, M, 3)
            feat = self.net.encode(x)
            rel = jt.ones((x.shape[0],))
            F = self.net.fuse(x, rel, feat, feat)
            score = self.net.predict_score(x, x, F)            # (b, M, 3)
            norms.append(jt.sqrt((score ** 2).sum(-1) + 1e-12).reshape(-1))
            i += step
        norms = jt.concat(norms, dim=0)
        return float(norms.var().item())

    @jt.no_grad()
    def _denoise_patches(self, patches_centered, tau, timesteps):
        """Iterative denoising of a batch of patches following the adaptive
        schedule (Algorithm 2, inner loop)."""
        step = self.predict_patch_step
        out = []
        i = 0
        P = patches_centered.shape[0]
        tau = max(tau, 1)
        while i < P:
            x = patches_centered[i:i + step]                   # (b, M, 3)
            feat_ref = self.net.encode(x)                      # E(x_tau), the reference
            x_cur = x
            for (t_cur, t_next) in timesteps:
                feat_cur = self.net.encode(x_cur)
                rel_t = jt.ones((x_cur.shape[0],)) * (float(t_cur) / float(tau))
                F = self.net.fuse(x_cur, rel_t, feat_cur, feat_ref)
                score = self.net.predict_score(x_cur, x_cur, F)
                coef = float(self.schedule.update_coef(t_cur, t_next))
                x_cur = x_cur + coef * score                   # Eq. 7
            out.append(x_cur)
            i += step
        return jt.concat(out, dim=0)

    @jt.no_grad()
    def adaptive_denoise(self, pcl_noisy):
        """Full Algorithm 2 for a single (N, 3) point cloud."""
        patches_c, seed_expand, point_idxs, best_patch, P = self._build_patches(pcl_noisy)

        sigma2 = self._estimate_noise_var(patches_c) * self.predict_noise_scale  # Eq. 15 (+scale)
        tau = self.schedule.find_tau(sigma2)                   # Eq. 16
        timesteps = self.schedule.adaptive_timesteps(tau, self.L)

        denoised_c = self._denoise_patches(patches_c, tau, timesteps)
        denoised = denoised_c + seed_expand                    # un-center -> (P, M, 3)

        # ---- stitch: each point takes the position from its closest patch ----
        N = pcl_noisy.shape[0]
        out = pcl_noisy.clone()
        for p in range(P):
            idxs = point_idxs[p]                               # (M,)
            sel = (best_patch[idxs] == p)
            if sel.sum().item() > 0:
                target = idxs[sel]
                out[target] = denoised[p][sel]
        return out, tau

    @staticmethod
    def smooth_outliers(points, k=16, n_sigma=3.0):
        """Pull statistical outliers toward their local neighbour mean."""
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        _, idx = tree.query(points, k=k + 1)
        neighbors = points[idx[:, 1:]]
        neighbor_mean = neighbors.mean(axis=1)
        dist_to_mean = np.linalg.norm(points - neighbor_mean, axis=1)
        avg = dist_to_mean.mean()
        mask = dist_to_mean > n_sigma * avg
        if mask.sum() > 0:
            alpha = np.clip(1.0 - (n_sigma * avg) / (dist_to_mean[mask] + 1e-8), 0, 1)
            points[mask] = points[mask] + alpha[:, None] * (neighbor_mean[mask] - points[mask])
        return points

    @staticmethod
    def pca_smooth(points, k=12, alpha=0.3):
        """Move each point slightly along its local normal toward the neighbourhood
        mean (mild surface smoothing)."""
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        _, idx = tree.query(points, k=k + 1)
        neighbors = points[idx[:, 1:]]
        neighbor_mean = neighbors.mean(axis=1)
        centered = neighbors - neighbor_mean[:, None, :]
        cov = np.einsum('nki,nkj->nij', centered, centered) / k
        _, eigvec = np.linalg.eigh(cov)
        normals = eigvec[:, :, 0]
        disp = neighbor_mean - points
        normal_comp = np.sum(disp * normals, axis=1, keepdims=True) * normals
        return points + alpha * normal_comp

    def post_process(self, points):
        points = self.smooth_outliers(points)
        if self.pca_smooth_alpha > 0:
            points = self.pca_smooth(points, alpha=self.pca_smooth_alpha)
        return points

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_denoised, tau = self.adaptive_denoise(pc_noisy)
            pc_denoised = pc_denoised.detach().numpy()
            if self.predict_post_smooth:
                pc_denoised = self.post_process(pc_denoised)

            if 'norm_center' in batch and 'norm_scale' in batch:
                center = batch['norm_center'][i]
                scale = batch['norm_scale'][i]
                if isinstance(center, jt.Var):
                    center = center.numpy()
                if isinstance(scale, jt.Var):
                    scale = scale.numpy()
                pc_denoised = pc_denoised * scale + center
            res.append({"pc_denoised": pc_denoised})
        return res
