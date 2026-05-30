import math
import os
import jittor as jt
from jittor import nn
from typing import Dict, List, Optional

import numpy as np

from .feature import FeatureExtraction
from .classify import ScaleNet
from .loss import calc_cd_like_InfoV2

from ..spec import ModelSpec
from ..feature import get_knn_idx


def get_entropy_B(point_clouds):
    """Compute entropy of point cloud distribution via voxelization.

    Matches official implementation exactly.
    Args:
        point_clouds: (B, N, 3)
    Returns:
        entropies: (B,)
    """
    entropies = []
    for point_cloud in point_clouds:
        n_pts = point_cloud.shape[0]
        voxel_size = 0.001 * 100000.0 / n_pts
        min_bound = point_cloud.min(dim=0, keepdims=False)
        max_bound = point_cloud.max(dim=0, keepdims=False)
        vc = jt.ceil((max_bound - min_bound) / voxel_size).int32()
        vc0, vc1 = vc[0].item(), vc[1].item()

        indices = jt.floor((point_cloud - min_bound) / voxel_size).int32()
        stride = jt.array([1, vc0, vc0 * vc1]).int32()
        hashed = (indices * stride).sum(dim=1)
        unique_hashes, inverse_indices = jt.unique(hashed, return_inverse=True)

        n_voxels = unique_hashes.shape[0]
        voxel_densities = jt.zeros((n_voxels,), dtype='float32')
        ones = jt.ones((inverse_indices.shape[0],), dtype='float32')
        voxel_densities = voxel_densities.scatter_(0, inverse_indices, ones, reduce='add')

        max_density = voxel_densities.max().item()
        if max_density <= 0:
            entropies.append(jt.array(0.0))
            continue

        hist = jt.zeros((10,), dtype='float32')
        for b in range(10):
            low = b * max_density / 10.0
            high = (b + 1) * max_density / 10.0
            mask = (voxel_densities >= low) & (voxel_densities < high)
            hist[b] = mask.float32().sum()

        hist = hist.maximum(jt.array(1.0))
        hist = hist / hist.sum()
        ent = -(hist * jt.log(hist + 1e-8)).sum()
        entropies.append(ent)

    return jt.stack(entropies, dim=0)


def farthest_point_sampling(pcls, num_pnts):
    """Farthest point sampling on batched point clouds.

    Args:
        pcls: (B, N, 3)
        num_pnts: number of points to sample
    Returns:
        sampled: (B, num_pnts, 3)
        indices: (B, num_pnts)
    """
    B, N, _ = pcls.shape
    sampled_list = []
    indices_list = []
    for b in range(B):
        pts = pcls[b]
        selected = []
        dist = jt.ones((N,)) * 1e10
        farthest = 0
        for _ in range(num_pnts):
            selected.append(farthest)
            centroid = pts[farthest]
            d = ((pts - centroid) ** 2).sum(dim=1)
            dist = jt.minimum(dist, d)
            farthest, _ = jt.argmax(dist, dim=-1)
            farthest = farthest.item()
        idx = jt.array(selected).int32()
        sampled_list.append(pts[idx].unsqueeze(0))
        indices_list.append(idx.unsqueeze(0))
    return jt.concat(sampled_list, dim=0), jt.concat(indices_list, dim=0)


class ClassifyModule(ModelSpec):
    """Module for training the ScaleNet classifier (Stage 1 of ASDN).

    Trains ScaleNet to predict entropy ratio rho = H(noise) / H(clean).
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        self.frame_knn = cfg.get('frame_knn', 32)
        self.classifier = ScaleNet(
            k=self.frame_knn,
            input_dim=3,
            z_dim=0,
            embedding_dim=cfg.get('classify_embedding_dim', 256),
            output_dim=1,
        )

    def training_step(self, batch: Dict) -> Dict:
        pc_noisy = batch['pc_noisy']
        pc_clean = batch['pc_clean']

        noisy_centered = pc_noisy - pc_noisy.mean(dim=1, keepdims=True)
        clean_centered = pc_clean - pc_clean.mean(dim=1, keepdims=True)

        noise_entropy = get_entropy_B(noisy_centered)
        clean_entropy = get_entropy_B(clean_centered)
        entropy_ratio = noise_entropy / (clean_entropy + 1e-8)

        pred_rho, _ = self.classifier(pc_noisy, None)
        # reshape(-1) instead of squeeze(): with batch_size=1 (validation) squeeze
        # would collapse (1,1)->0-D, which Jittor's reshape rejects.
        loss = ((pred_rho.reshape(-1) - entropy_ratio.reshape(-1)) ** 2).sum()

        return {'loss': loss, 'mse': loss}

    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)

    def process_fn(self, batch: List) -> List[Dict]:
        res = []
        for b in batch:
            assert b.meta is not None
            res.append({
                'pc_noisy': np.asarray(b.meta['pc_noisy'][0], dtype=np.float32),
                'pc_clean': np.asarray(b.meta['pc_clean'][0], dtype=np.float32),
            })
        return res

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy = batch['pc_noisy']
        pred_rho, _ = self.classifier(pc_noisy, None)
        result = []
        for i in range(pc_noisy.shape[0]):
            result.append({'rho': pred_rho[i].item()})
        return result


class ASDNModule(ModelSpec):
    """ASDN Denoising Module (Stage 2).

    Uses a frozen ScaleNet classifier to guide adaptive stopping
    in the FeatureExtraction U-Net for point cloud denoising.
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        self.frame_knn = cfg.get('frame_knn', 32)
        self.num_train_points = cfg.get('num_train_points', 1000)

        self.feature_nets = FeatureExtraction(
            d_in=0,
            d_out=cfg.get('feat_d_out', 32),
            n_cls=3,
            nsample=cfg.get('nsample', 16),
            stride_list=cfg.get('stride_list', [4, 3, 2, 1]),
        )

        self.classifier = ScaleNet(
            k=self.frame_knn,
            input_dim=3,
            z_dim=0,
            embedding_dim=cfg.get('classify_embedding_dim', 256),
            output_dim=1,
        )
        self.feature_nets.set_classifier(self.classifier)

        # Stage 2 (ASDN denoiser) relies on the Stage 1 classifier to choose the
        # per-sample stopping layer. Load the pretrained ScaleNet weights from a
        # ClassifyModule checkpoint if provided; the classifier is never updated
        # during Stage 2 (the ceil() in layer assignment blocks gradients), so it
        # stays effectively frozen.
        classify_ckpt = cfg.get('classify_ckpt', None)
        if classify_ckpt is not None:
            self.set_classifier_weights(classify_ckpt)

    def set_classifier_weights(self, classify_ckpt: str):
        """Load pretrained classifier weights from a stage-1 ClassifyModule checkpoint."""
        if not os.path.exists(classify_ckpt):
            print(f"\033[31m[ASDN] classify_ckpt not found: {classify_ckpt} -> "
                  f"using randomly-initialized classifier\033[0m")
            return
        sd = jt.load(classify_ckpt)
        # ClassifyModule stores ScaleNet under the 'classifier.' prefix, matching
        # ASDNModule.classifier, so we can filter and load by relative name.
        cls_sd = {k[len('classifier.'):]: v for k, v in sd.items()
                  if k.startswith('classifier.')}
        if not cls_sd:
            print(f"\033[31m[ASDN] no 'classifier.*' params in {classify_ckpt}\033[0m")
            return
        self.classifier.load_parameters(cls_sd)
        print(f"\033[92m[ASDN] loaded pretrained classifier from {classify_ckpt} "
              f"({len(cls_sd)} tensors)\033[0m")

    def training_step(self, batch: Dict) -> Dict:
        pc_noisy = batch['pc_noisy']
        pc_clean = batch['pc_clean']

        B, N_noisy, _ = pc_noisy.shape
        N_clean = pc_clean.shape[1]

        pcl_seeds, _ = farthest_point_sampling(pc_noisy, 1)

        pcl_seeds_expand_noisy = pcl_seeds.broadcast((B, N_noisy, 3))
        pcl_input = pc_noisy - pcl_seeds_expand_noisy

        pcl_seeds_expand_clean = pcl_seeds.broadcast((B, N_clean, 3))
        pcl_clean_centered = pc_clean - pcl_seeds_expand_clean

        offset = jt.array([(i + 1) * N_noisy for i in range(B)]).int32()

        feat = pcl_input.reshape(B * N_noisy, -1)[:, 3:]
        pred_disp = self.feature_nets(pcl_input, feat, offset, classify_net=self.classifier)

        pred_pcl = pcl_input + pred_disp

        loss = calc_cd_like_InfoV2(pred_pcl, pcl_clean_centered)
        loss = loss.mean()

        return {'loss': loss}

    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)

    def process_fn(self, batch: List) -> List[Dict]:
        res = []
        if self.is_predict():
            for b in batch:
                assert b.meta is not None
                res.append({
                    'pc_noisy': np.asarray(b.sampled_vertices_noisy, dtype=np.float32),
                    'norm_center': np.asarray(b.meta['norm_center'], dtype=np.float32).reshape(3),
                    'norm_scale': np.asarray(b.meta['norm_scale'], dtype=np.float32).reshape(1),
                })
        else:
            for b in batch:
                assert b.meta is not None
                res.append({
                    'pc_noisy': np.asarray(b.meta['pc_noisy'][0], dtype=np.float32),
                    'pc_clean': np.asarray(b.meta['pc_clean'][0], dtype=np.float32),
                })
        return res

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps=1):
        """Single-step denoising (ASDN does one forward pass, not iterative)."""
        B, N, d = pcl_noisy.shape

        with jt.no_grad():
            offset = jt.array([(i + 1) * N for i in range(B)]).int32()
            feat = pcl_noisy.reshape(B * N, -1)[:, 3:]
            pred_disp = self.feature_nets(pcl_noisy, feat, offset, classify_net=self.classifier)

        return pcl_noisy + pred_disp

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3

        patch_size = self.model_config.get('predict_patch_size', 1000)

        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_denoised = patch_based_denoise_asdn(
                model=self,
                pcl_noisy=pc_noisy,
                patch_size=patch_size,
                seed_k=self.model_config.get('predict_seed_k', 5),
                seed_k_alpha=self.model_config.get('predict_seed_k_alpha', 10),
            )

            if isinstance(pc_denoised, jt.Var):
                pc_denoised = pc_denoised.numpy()

            pc_denoised = self._smooth_outliers(pc_denoised)

            if 'norm_center' in batch and 'norm_scale' in batch:
                center = batch['norm_center'][i]
                scale = batch['norm_scale'][i]
                if isinstance(center, jt.Var):
                    center = center.numpy()
                if isinstance(scale, jt.Var):
                    scale = scale.numpy()
                pc_denoised = pc_denoised * scale + center

            res.append({'pc_denoised': pc_denoised})
        return res

    @staticmethod
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


def patch_based_denoise_asdn(model, pcl_noisy, patch_size=1000, seed_k=6, seed_k_alpha=1):
    """Patch-based denoising with patch stitching, matching official implementation."""
    assert pcl_noisy.ndim == 2
    N, d = pcl_noisy.shape
    pcl_noisy_b = pcl_noisy.unsqueeze(0)

    num_patches = int(seed_k * N / patch_size)
    seed_pnts, _ = farthest_point_sampling(pcl_noisy_b, num_patches)
    _, point_idxs, patches = knn_points(seed_pnts, pcl_noisy_b, patch_size)

    patches = patches[0]
    point_idxs = point_idxs[0]

    seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
    patches_centered = patches - seed_expand

    patch_dists = ((patches - seed_expand) ** 2).sum(dim=-1)
    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)

    all_dists = jt.ones((num_patches, N)) * 1e10
    all_dists = all_dists.scatter_(1, point_idxs, patch_dists)

    weights = jt.exp(-all_dists)
    best_weights_idx, _ = jt.argmax(weights, dim=0)

    patches_denoised_list = []
    i = 0
    patch_step = max(1, int(math.ceil(N / (seed_k_alpha * patch_size))))
    while i < num_patches:
        curr = patches_centered[i:i + patch_step]
        try:
            out = model.denoise_langevin_dynamics(curr)
        except Exception as e:
            print(f"Denoise error: {e}")
            return None
        patches_denoised_list.append(out)
        i += patch_step

    patches_denoised = jt.concat(patches_denoised_list, dim=0)
    patches_denoised = patches_denoised + seed_expand

    # Vectorized patch stitching: for each global point, keep the denoised
    # coordinate coming from its highest-weight (closest-seed) patch. Points not
    # covered by any patch fall back to their noisy coordinate. Equivalent to the
    # official per-point loop but without N host syncs.
    flat_global = point_idxs.reshape(-1)                          # (P*K,)
    flat_patch = jt.arange(num_patches).reshape(num_patches, 1)
    flat_patch = flat_patch.broadcast((num_patches, patch_size)).reshape(-1)
    flat_denoised = patches_denoised.reshape(-1, 3)               # (P*K, 3)
    chosen = best_weights_idx[flat_global]                        # best patch per occurrence
    keep = (chosen == flat_patch)
    pcl_out = pcl_noisy_b[0].clone()                              # fallback: noisy
    pcl_out[flat_global[keep]] = flat_denoised[keep]
    return pcl_out


def knn_points(x, y, k):
    """Batched KNN."""
    _, idx = jt.misc.knn(x, y, k)
    B, P, _ = x.shape
    _, N, _ = y.shape

    nn_list = []
    for b in range(B):
        nn_list.append(y[b][idx[b]])
    nn = jt.stack(nn_list, dim=0)

    dist = ((x.unsqueeze(2) - nn) ** 2).sum(dim=-1)
    return dist, idx, nn
