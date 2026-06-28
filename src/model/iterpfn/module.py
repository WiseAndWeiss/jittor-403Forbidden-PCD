"""IterativePFN-style stacked iterative point-cloud filtering on the ASDN backbone.

Two schedules (cfg `schedule`):
  - 'refine' (default): every module is supervised toward the CLEAN cloud. Module
    1 reproduces v2's objective; modules 2..T see ALREADY-DENOISED inputs (detached)
    so they learn small corrections, NOT re-denoising -> doesn't over-smooth the way
    naive same-net iteration did. Low downside (>= v2 if refiners learn identity).
  - 'progressive': module t targets clean + alpha_t * noise with alpha_t=(T-t)/T,
    the faithful IterativePFN decreasing-noise trajectory.
Inputs are detached between modules.
"""

import os
import math
import jittor as jt
from jittor import nn
from typing import Dict, List

import numpy as np

from ..asdn.feature import FeatureExtraction
from ..asdn.loss import calc_cd_like_InfoV2
from ..asdn.module import patch_based_denoise_asdn, farthest_point_sampling
from ..spec import ModelSpec


class IterativePFNModule(ModelSpec):

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        self.num_iters = int(cfg.get('num_iters', 3))
        self.schedule = cfg.get('schedule', 'refine')   # 'refine' | 'progressive'
        self.loss_type = cfg.get('loss_type', 'infocd')  # 'infocd' | 'mse'
        # Train end-to-end through the module stack.
        self.detach_between = bool(cfg.get('detach_between', False))
        self.feat_d_out = cfg.get('feat_d_out', 32)

        self.iter_modules = nn.ModuleList([
            FeatureExtraction(
                d_in=0,
                d_out=self.feat_d_out,
                n_cls=3,
                nsample=cfg.get('nsample', 16),
                stride_list=cfg.get('stride_list', [4, 3, 2, 1]),
            )
            for _ in range(self.num_iters)
        ])
        # No adaptive classifier: each module runs fixed full depth (classify_net=None
        # -> n_layers=[4]*B inside FeatureExtraction).
        for m in self.iter_modules:
            m.set_classifier(None)

        init_ckpt = cfg.get('init_ckpt', None)
        if init_ckpt is not None:
            self._warmstart_from(init_ckpt)

    def _warmstart_from(self, ckpt_path: str):
        """Init every iteration module from a trained ASDNModule's feature_nets."""
        if not os.path.exists(ckpt_path):
            print(f"\033[31m[IterPFN] init_ckpt not found: {ckpt_path} -> random init\033[0m")
            return
        sd = jt.load(ckpt_path)
        feat_sd = {k[len('feature_nets.'):]: v for k, v in sd.items()
                   if k.startswith('feature_nets.')}
        if not feat_sd:
            print(f"\033[31m[IterPFN] no 'feature_nets.*' in {ckpt_path}\033[0m")
            return
        for i, m in enumerate(self.iter_modules):
            # Clone per module: loading the SAME Var objects into every module
            # aliases their parameters, so a single backward over the summed loss
            # routes all gradient to module 0 and the rest never update. Distinct
            # cloned Vars keep each module independently trainable.
            sd_copy = {k: v.clone() for k, v in feat_sd.items()}
            m.load_parameters(sd_copy)
        print(f"\033[92m[IterPFN] warm-started {self.num_iters} modules from {ckpt_path} "
              f"({len(feat_sd)} tensors each, cloned per module)\033[0m")

    # ---- helpers ----
    def _run_module(self, module, pc):
        """One iteration step: returns refined cloud pc + predicted displacement.
        pc: (B, N, 3)."""
        B, N, _ = pc.shape
        offset = jt.array([(i + 1) * N for i in range(B)]).int32()
        feat = pc.reshape(B * N, -1)[:, 3:]
        disp = module(pc, feat, offset, classify_net=None)
        return pc + disp

    def _step_loss(self, pred, target):
        if self.loss_type == 'mse':
            return ((pred - target) ** 2).sum(dim=-1).mean()
        return calc_cd_like_InfoV2(pred, target)

    # ---- training ----
    def training_step(self, batch: Dict) -> Dict:
        pc_noisy = batch['pc_noisy']     # (B, N, 3) seed-centered
        pc_clean = batch['pc_clean']     # (B, N, 3) seed-centered, point-corresponded
        B, N, _ = pc_noisy.shape
        T = self.num_iters
        noise = pc_noisy - pc_clean      # per-point noise (correspondence from AugmentPatch)

        total = 0.0
        comps = {}
        p_prev = pc_noisy
        for t in range(1, T + 1):
            if self.schedule == 'progressive':
                alpha = (T - t) / float(T)
                target_t = pc_clean + alpha * noise
            else:  # 'refine' -> always clean
                target_t = pc_clean
            inp = p_prev.detach() if self.detach_between else p_prev
            p_t = self._run_module(self.iter_modules[t - 1], inp)
            loss_t = self._step_loss(p_t, target_t)
            total = total + loss_t
            comps[f'l{t}'] = loss_t
            p_prev = p_t

        out = {'loss': total}
        out.update(comps)
        return out

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

    # ---- inference ----
    def denoise_langevin_dynamics(self, pcl_noisy, num_steps=1):
        """Run all T iteration modules in sequence on a batch of patches.
        pcl_noisy: (P, K, 3) centered patches."""
        x = pcl_noisy
        with jt.no_grad():
            for m in self.iter_modules:
                x = self._run_module(m, x)
        return x

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3
        patch_size = self.model_config.get('predict_patch_size', 1000)
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            den = patch_based_denoise_asdn(
                model=self,
                pcl_noisy=pc_noisy,
                patch_size=patch_size,
                seed_k=self.model_config.get('predict_seed_k', 5),
                seed_k_alpha=self.model_config.get('predict_seed_k_alpha', 10),
                stitch=self.model_config.get('predict_stitch', 'best'),
            )
            if not isinstance(den, jt.Var):
                den = jt.array(den)
            pc_denoised = den.numpy()
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
