from typing import Dict, List

import jittor as jt
import numpy as np

from .feature import FeatureExtraction, DecoderStraight
from .spec import ModelSpec

from ..data.asset import Asset


def get_random_indices(n, m):
    assert m <= n
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()


class StraightVelocityModule(ModelSpec):

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.frame_knn = cfg['frame_knn']
        self.num_train_points = cfg['num_train_points']
        self.dsm_sigma = cfg['dsm_sigma']

        self.cd_loss_weight = cfg.get('cd_loss_weight', 0.0)
        self.num_cd_points = cfg.get('num_cd_points', 200)
        self.smoothness_weight = cfg.get('smoothness_weight', 0.0)
        self.smoothness_k = cfg.get('smoothness_k', 8)

        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg['feat_embedding_dim'],
        )

        self.decoder = DecoderStraight(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg['decoder_hidden_dim'],
        )

    def get_supervised_loss(self, pc_noisy_L2, pc_mix, pc_clean):
        """
        X_0 = pc_noisy_L2: original noisy reference (B, N, 3)
        X_t = pc_mix:       intermediate state (B, N, 3)
        X_1 = pc_clean:     clean (B, N, 3)

        Target: X_1 - X_0 (constant, independent of t)
        """
        B, N_noisy, d = pc_mix.shape

        pnt_idx = get_random_indices(N_noisy, self.num_train_points)

        feat = self.encoder(pc_mix)
        F_dim = feat.shape[2]

        feat = feat[:, pnt_idx, :]
        pc_noisy_L2 = pc_noisy_L2[:, pnt_idx, :]
        pc_mix = pc_mix[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]

        grad_dir_t_target = pc_clean - pc_noisy_L2

        pred_dir = self.decoder(
            c=feat.reshape(-1, F_dim),
        ).reshape(B, len(pnt_idx), d)

        mse_loss = (((pred_dir - grad_dir_t_target) ** 2.0) / self.dsm_sigma).sum(dim=-1).mean()

        if self.smoothness_weight > 0:
            M = len(pnt_idx)
            pred_dir_reshaped = pred_dir.reshape(B, M, d)
            dist = ((pc_mix.unsqueeze(2) - pc_mix.unsqueeze(1)) ** 2).sum(dim=-1)
            sk = min(self.smoothness_k, M - 1)
            _, nn_idx = jt.topk(dist, k=sk + 1, dim=-1, largest=False)
            nn_idx = nn_idx[:, :, 1:]
            neighbor_disps = []
            for b in range(B):
                neighbor_disps.append(pred_dir_reshaped[b][nn_idx[b]])
            neighbor_disps = jt.stack(neighbor_disps, dim=0)
            neighbor_mean = neighbor_disps.mean(dim=2)
            smoothness_loss = ((pred_dir_reshaped - neighbor_mean) ** 2).sum(dim=-1).mean()
        else:
            smoothness_loss = jt.array(0.0)

        pc_pred = pc_mix + pred_dir

        cd_loss = jt.array(0.0)
        if self.cd_loss_weight > 0:
            cd_idx = get_random_indices(len(pnt_idx), self.num_cd_points)
            pc_pred_cd = pc_pred[:, cd_idx, :]
            pc_clean_cd = pc_clean[:, cd_idx, :]
            cd_loss = self._compute_cd(pc_pred_cd, pc_clean_cd)

        loss = mse_loss + self.smoothness_weight * smoothness_loss + self.cd_loss_weight * cd_loss

        return loss, mse_loss, cd_loss, smoothness_loss

    def _compute_cd(self, pc_a, pc_b):
        B, M, _ = pc_a.shape
        diff = pc_a.unsqueeze(2) - pc_b.unsqueeze(1)
        dist = (diff ** 2).sum(dim=-1)
        cd_ab = dist.min(dim=-1)[0].mean()
        cd_ba = dist.min(dim=-2)[0].mean()
        return cd_ab + cd_ba

    def denoise_straight(self, pcl_noisy):
        """
        pcl_noisy: (B, N, 3)
        Single-step straight filtering。
        """
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            feat = self.encoder(pcl_noisy)
            F_dim = feat.shape[2]
            pred_dir = self.decoder(
                c=feat.reshape(-1, F_dim),
            ).reshape(B, N, d)
        return pcl_noisy + pred_dir, None

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy_L2'].shape[-2]
        pc_noisy_L2 = batch['pc_noisy_L2'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        loss, mse_loss, cd_loss, smoothness_loss = self.get_supervised_loss(
            pc_noisy_L2=pc_noisy_L2,
            pc_mix=pc_mix,
            pc_clean=pc_clean,
        )
        return {"loss": loss, "mse_loss": mse_loss, "cd_loss": cd_loss, "smoothness_loss": smoothness_loss}

    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3

        num_steps = self.model_config.get('predict_num_steps', 1)
        patch_size = self.model_config.get('predict_patch_size', 2048)

        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_denoised = patch_based_denoise_straight(
                model=self,
                pcl_noisy=pc_noisy,
                patch_size=patch_size,
                seed_k=6,
                seed_k_alpha=1,
                num_steps=num_steps,
            ).detach().numpy()

            pc_denoised = self._smooth_outliers(pc_denoised)

            bilateral_alpha = self.model_config.get('bilateral_smooth_alpha', 1.0)
            if bilateral_alpha > 0:
                pc_denoised = self._bilateral_smooth(pc_denoised)

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

    @staticmethod
    def _smooth_outliers(points, k=16, n_sigma=2.5):
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        _, idx = tree.query(points, k=k + 1)
        neighbors = points[idx[:, 1:]]
        neighbor_mean = neighbors.mean(axis=1)
        dist_to_mean = np.linalg.norm(points - neighbor_mean, axis=1)
        avg_dist = dist_to_mean.mean()
        mask = dist_to_mean > n_sigma * avg_dist
        if mask.sum() > 0:
            alpha = np.clip(1.0 - (n_sigma * avg_dist) / (dist_to_mean[mask] + 1e-8), 0, 1)
            points[mask] = points[mask] + alpha[:, None] * (neighbor_mean[mask] - points[mask])
        return points

    @staticmethod
    def _estimate_normals(points, k=24):
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        _, idx = tree.query(points, k=k + 1)
        neighbors = points[idx[:, 1:]]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        cov = np.einsum('nki,nkj->nij', centered, centered) / k
        _, eigenvectors = np.linalg.eigh(cov)
        return eigenvectors[:, :, 0]

    @staticmethod
    def _bilateral_smooth(points, sigma_s=0.03, sigma_c=0.3, k=24):
        from scipy.spatial import cKDTree
        normals = StraightVelocityModule._estimate_normals(points, k=k)
        tree = cKDTree(points)
        dists, idx = tree.query(points, k=k + 1)
        neighbor_pts = points[idx[:, 1:]]
        neighbor_normals = normals[idx[:, 1:]]
        displacements = neighbor_pts - points[:, None, :]
        normal_projections = np.sum(displacements * normals[:, None, :], axis=-1)
        w_s = np.exp(-0.5 * (dists[:, 1:] / sigma_s) ** 2)
        normal_diffs = neighbor_normals - normals[:, None, :]
        n_dist = np.sum(normal_diffs ** 2, axis=-1)
        w_c = np.exp(-0.5 * n_dist / (sigma_c ** 2 + 1e-8))
        weights = w_s * w_c
        weight_sum = weights.sum(axis=1, keepdims=True) + 1e-8
        weighted_proj = (weights * normal_projections).sum(axis=1, keepdims=True) / weight_sum
        return points + weighted_proj * normals

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({
                    "pc_noisy_L2": b.meta['pc_noisy_L2'],
                    "pc_clean": b.meta['pc_clean'],
                    "pc_mix": b.meta['pc_mix'],
                })
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy,
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                if b.meta is not None:
                    d["norm_center"] = np.array(b.meta.get('norm_center'))
                    d["norm_scale"] = np.array(b.meta.get('norm_scale'))
                res.append(d)
        return res


def farthest_point_sampling(pcls, num_pnts):
    B, N, _ = pcls.shape
    sampled = []
    indices = []
    for b in range(B):
        pts = pcls[b]
        selected = []
        dist = jt.ones((N,)) * 1e10
        farthest = 0
        for i in range(num_pnts):
            selected.append(farthest)
            centroid = pts[farthest]
            d = ((pts - centroid) ** 2).sum(dim=1)
            dist = jt.minimum(dist, d)
            farthest, _ = jt.argmax(dist, dim=-1)
            farthest = farthest.item()
        idx = jt.array(selected).int32()
        sampled.append(pts[idx][None, ...])
        indices.append(idx[None, ...])
    sampled = jt.concat(sampled, dim=0)
    indices = jt.concat(indices, dim=0)
    return sampled, indices


def knn_points(x, y, k):
    dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    dist_k, idx = jt.topk(dist, k=k, dim=-1, largest=False)
    B = x.shape[0]
    nn = []
    for b in range(B):
        nn.append(y[b][idx[b]])
    nn = jt.stack(nn, dim=0)
    return dist_k, idx, nn


from math import ceil


def patch_based_denoise_straight(model: StraightVelocityModule, pcl_noisy,
                                  patch_size=2048, seed_k=6, seed_k_alpha=1,
                                  num_steps=1) -> jt.Var:
    assert len(pcl_noisy.shape) == 2

    N, d = pcl_noisy.shape
    if num_steps <= 1:
        pcl_noisy = pcl_noisy.unsqueeze(0)
        out, _ = model.denoise_straight(pcl_noisy)
        return out[0]

    num_patches = int(seed_k * N / patch_size)
    pcl_noisy = pcl_noisy.unsqueeze(0)

    seed_pnts, seed_idx = farthest_point_sampling(pcl_noisy, num_patches)
    _, point_idxs, patches = knn_points(seed_pnts, pcl_noisy, patch_size)

    patches = patches[0]
    point_idxs = point_idxs[0]

    seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expand

    patch_dists = ((seed_pnts.squeeze().unsqueeze(1) - pcl_noisy) ** 2).sum(dim=-1)
    patch_dists = patch_dists[0]
    all_dists = jt.ones((num_patches, N)) * 1e10
    for i in range(num_patches):
        all_dists[i][point_idxs[i]] = patch_dists[i]

    weights = jt.exp(-all_dists)
    best_weights_idx, _ = jt.argmax(weights, dim=0)

    patch_step = int(ceil(N / (seed_k_alpha * patch_size)))
    assert patch_step > 0

    pcl_next = patches.clone()
    for _step in range(num_steps):
        patches_denoised = []
        i = 0
        while i < num_patches:
            curr = pcl_next[i:i + patch_step]
            out, _ = model.denoise_straight(curr)
            patches_denoised.append(out)
            i += patch_step
        pcl_next = jt.concat(patches_denoised, dim=0)

    patches_denoised = pcl_next + seed_expand
    pcl_out = []
    for pidx in range(N):
        patch_id = best_weights_idx[pidx].item()
        mask = (point_idxs[patch_id] == pidx)
        if mask.sum().item() > 0:
            pcl_out.append(patches_denoised[patch_id][mask])
        else:
            pcl_out.append(pcl_noisy[0, pidx:pidx + 1])
    pcl_out = jt.concat(pcl_out, dim=0)
    return pcl_out
