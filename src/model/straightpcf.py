from typing import Dict, List

import jittor as jt
from jittor import nn
import numpy as np

from .feature import FeatureExtraction, DistanceDecoder, DecoderStraight
from .spec import ModelSpec
from .straight_vm import get_random_indices

from ..data.asset import Asset


class StraightPCF(ModelSpec):

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        self.frame_knn = cfg['frame_knn']
        self.tot_its = cfg.get('predict_num_steps', 4)
        self.num_train_points = cfg['num_train_points']
        self.dsm_sigma = cfg['dsm_sigma']

        self.num_modules = cfg.get('num_modules', 2)
        self.finetune_loss_weight = cfg.get('finetune_loss_weight', 1.0)
        self.cd_loss_weight = cfg.get('cd_loss_weight', 0.0)
        self.num_cd_points = cfg.get('num_cd_points', 200)
        self.smoothness_weight = cfg.get('smoothness_weight', 0.0)
        self.smoothness_k = cfg.get('smoothness_k', 8)

        self.velocity_nets = nn.ModuleList()
        for _i in range(self.num_modules):
            self.velocity_nets.append(nn.Sequential(
                FeatureExtraction(
                    k=self.frame_knn,
                    input_dim=3,
                    embedding_dim=cfg['feat_embedding_dim'],
                ),
                DecoderStraight(
                    z_dim=cfg['feat_embedding_dim'],
                    dim=3,
                    out_dim=3,
                    hidden_size=cfg['decoder_hidden_dim'],
                ),
            ))

        self.distance_encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg['feat_embedding_dim'],
            distance_estimation=True,
        )

        self.distance_decoder = DistanceDecoder(
            z_dim=self.distance_encoder.embedding_dim,
            hidden_size=cfg['decoder_hidden_dim'],
        )

    def get_supervised_loss(self, pc_noisy_L2, pc_mix, pc_clean):
        B, N, d = pc_mix.shape
        pnt_idx = get_random_indices(N, self.num_train_points)

        pc_noisy_L2 = pc_noisy_L2[:, pnt_idx, :]
        pc_mix = pc_mix[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]

        num = jt.sqrt(((pc_clean - pc_mix) ** 2).sum(dim=-1))
        den = jt.sqrt(((pc_clean - pc_noisy_L2) ** 2).sum(dim=-1))
        ratio_target = (num / (den + 1e-8)).mean(dim=1)

        feat_d = self.distance_encoder(pc_mix)
        F_d = feat_d.shape[2]
        pred_d = self.distance_decoder(
            c=feat_d.reshape(-1, F_d), B=B, N=len(pnt_idx)
        ).reshape(B)

        loss_d = ((pred_d - ratio_target) ** 2).mean()

        M = len(pnt_idx)
        pcl_cur = pc_mix.clone()
        for mod_idx in range(self.num_modules):
            feat = self.velocity_nets[mod_idx][0](pcl_cur)
            F = feat.shape[2]
            pred_dir = self.velocity_nets[mod_idx][1](
                c=feat.reshape(-1, F)
            ).reshape(B, M, d)
            pcl_cur = pcl_cur + (1.0 / self.num_modules) * pred_d.reshape(B, 1, 1) * pred_dir

        finetune_loss = ((pc_clean - pcl_cur) ** 2).sum(dim=-1).mean()

        loss = (loss_d + self.finetune_loss_weight * finetune_loss) / self.dsm_sigma

        return loss, loss_d, finetune_loss

    def denoise_straightpcf(self, pcl_noisy):
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()

            feat_d = self.distance_encoder(pcl_next)
            F_d = feat_d.shape[2]
            pred_d = self.distance_decoder(
                c=feat_d.reshape(-1, F_d), B=B, N=N
            ).reshape(B, 1, 1)

            for it in range(self.tot_its):
                pred_disp = jt.zeros(B, N, d)
                for mod_idx in range(self.num_modules):
                    feat = self.velocity_nets[mod_idx][0](pcl_next)
                    F = feat.shape[2]
                    pred_dir = self.velocity_nets[mod_idx][1](
                        c=feat.reshape(-1, F)
                    ).reshape(B, N, d)
                    pred_disp = pred_disp + (1.0 / self.tot_its) * (1.0 / self.num_modules) * pred_d * pred_dir
                pcl_next = pcl_next + pred_disp
        return pcl_next, None

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy_L2'].shape[-2]
        pc_noisy_L2 = batch['pc_noisy_L2'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)

        loss, loss_d, finetune_loss = self.get_supervised_loss(
            pc_noisy_L2=pc_noisy_L2,
            pc_mix=pc_mix,
            pc_clean=pc_clean,
        )
        return {"loss": loss, "loss_d": loss_d, "finetune_loss": finetune_loss}

    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3

        patch_size = self.model_config.get('predict_patch_size', 2048)
        niters = self.model_config.get('predict_niters', 1)

        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_denoised = patch_based_denoise_straightpcf(
                model=self,
                pcl_noisy=pc_noisy,
                patch_size=patch_size,
                seed_k=6,
                seed_k_alpha=1,
                niters=niters,
            ).detach().numpy()

            pc_denoised = _smooth_outliers(
                pc_denoised,
                n_sigma=self.model_config.get('outlier_n_sigma', 2.0),
            )

            bilateral_alpha = self.model_config.get('bilateral_smooth_alpha', 1.0)
            if bilateral_alpha > 0:
                pc_denoised = _bilateral_smooth(
                    pc_denoised,
                    sigma_s=self.model_config.get('bilateral_sigma_s', 0.015),
                    sigma_c=self.model_config.get('bilateral_sigma_c', 0.2),
                    k=self.model_config.get('bilateral_k', 24),
                )

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


def _smooth_outliers(points, k=16, n_sigma=3.0):
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


def _estimate_normals(points, k=24):
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)
    neighbors = points[idx[:, 1:]]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', centered, centered) / k
    _, eigenvectors = np.linalg.eigh(cov)
    return eigenvectors[:, :, 0]


def _bilateral_smooth(points, sigma_s=0.03, sigma_c=0.3, k=24):
    from scipy.spatial import cKDTree
    normals = _estimate_normals(points, k=k)
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


from math import ceil


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


def patch_based_denoise_straightpcf(model: StraightPCF, pcl_noisy,
                                      patch_size=2048, seed_k=6, seed_k_alpha=1,
                                      niters=1) -> jt.Var:
    assert len(pcl_noisy.shape) == 2

    pcl_current = pcl_noisy
    for _round in range(niters):
        N, d = pcl_current.shape
        num_patches = int(seed_k * N / patch_size)
        pcl_input = pcl_current.unsqueeze(0)

        seed_pnts, seed_idx = farthest_point_sampling(pcl_input, num_patches)
        _, point_idxs, patches = knn_points(seed_pnts, pcl_input, patch_size)

        patches = patches[0]
        point_idxs = point_idxs[0]

        seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
        patches = patches - seed_expand

        patch_dists = ((seed_pnts.squeeze().unsqueeze(1) - pcl_input) ** 2).sum(dim=-1)
        patch_dists = patch_dists[0]
        all_dists = jt.ones((num_patches, N)) * 1e10
        for i in range(num_patches):
            all_dists[i][point_idxs[i]] = patch_dists[i]

        weights = jt.exp(-all_dists)
        best_weights_idx, _ = jt.argmax(weights, dim=0)

        max_patches_per_batch = 50
        patches_denoised_list = []
        i = 0
        while i < num_patches:
            chunk = patches[i:i + max_patches_per_batch]
            out_chunk, _ = model.denoise_straightpcf(chunk)
            patches_denoised_list.append(out_chunk)
            i += max_patches_per_batch
        patches_denoised = jt.concat(patches_denoised_list, dim=0)

        patches_denoised = patches_denoised + seed_expand

        pcl_out_list = []
        for pidx in range(N):
            patch_id = best_weights_idx[pidx].item()
            mask = (point_idxs[patch_id] == pidx)
            if mask.sum().item() > 0:
                pcl_out_list.append(patches_denoised[patch_id][mask])
            else:
                pcl_out_list.append(pcl_input[0, pidx:pidx + 1])
        pcl_current = jt.concat(pcl_out_list, dim=0)

    return pcl_current
