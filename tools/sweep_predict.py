#!/usr/bin/env python3
"""Quick predict-side hyper-parameter sweep on a few local-eval shapes.

Loads a trained SDN checkpoint and denoises a handful of shapes from the local
val set under different inference settings (noise-estimate scale, iteration count
L), scoring each with the same CD/P2S metric as evaluate.py. Lets us tune the
*inference* without retraining or running the full 32-shape eval each time.

    python tools/sweep_predict.py --ckpt experiments/sdn/checkpoint_15.pkl \
        --n 6 --scales 1,2,3,5 --L 5
"""
import argparse
import os
import sys
import time

import numpy as np
import jittor as jt
jt.flags.use_cuda = 1

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf
from src.model.parse import get_model
import evaluate as ev


def load_model(ckpt):
    mcfg = OmegaConf.to_container(OmegaConf.load('configs/model/sdn.yaml'), resolve=True)
    tcfg = OmegaConf.to_container(OmegaConf.load('configs/transform/sdn.yaml'), resolve=True)
    model = get_model(model_config=mcfg, transform_config=tcfg)
    model.load(ckpt)
    model.set_predict(True)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--list', default='datalist/local_test.txt')
    ap.add_argument('--mesh_dir', default='dataset_train')
    ap.add_argument('--eval_dir', default='local_eval')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--scale', type=float, default=1.0, help='noise estimate scale')
    ap.add_argument('--pca_alphas', default='0,0.2,0.3,0.5',
                    help='post-proc pca smoothing strengths to sweep (0 = raw)')
    ap.add_argument('--L', type=int, default=0, help='override num_iters (0=keep)')
    args = ap.parse_args()

    keys = [l.strip() for l in open(args.list) if l.strip()][:args.n]
    alphas = [float(a) for a in args.pca_alphas.split(',')]
    model = load_model(args.ckpt)
    model.predict_noise_scale = args.scale
    if args.L > 0:
        model.L = args.L
    print(f"[sweep] ckpt={args.ckpt} n={len(keys)} scale={args.scale} "
          f"L={model.L} pca_alphas={alphas}")

    # preload shapes and denoise each ONCE (raw); post-proc is applied afterwards.
    shapes = []
    t0 = time.time()
    for k in keys:
        noisy = np.load(os.path.join(args.eval_dir, 'noisy', k, 'noisy.npy')).astype(np.float64)
        gt = np.load(os.path.join(args.eval_dir, 'gt', k, 'clean.npy')).astype(np.float64)
        mv, mf = ev.load_mesh_vf(os.path.join(args.mesh_dir, k, 'models/model_normalized.obj'))
        den, tau = model.adaptive_denoise(jt.array(noisy.astype(np.float32)))
        den = den.detach().numpy().astype(np.float64)
        cd_noisy = ev.chamfer_distance(noisy, gt, normalize=True)
        p2s_noisy = (ev.point_to_surface_distance(noisy, mv, mf, normalize_ref_pc=gt)
                     if mv is not None else None)
        shapes.append((k, den, gt, mv, mf, cd_noisy, p2s_noisy))
    print(f"[sweep] denoised {len(keys)} shapes in {time.time()-t0:.1f}s")

    print(f"{'pca_a':>6} | {'CD_score':>8} | {'P2S_score':>9} | {'final':>6}")
    for a in alphas:
        cd_s, p2s_s = [], []
        for (k, den_raw, gt, mv, mf, cd_noisy, p2s_noisy) in shapes:
            den = den_raw.copy()
            if a > 0:
                den = model.smooth_outliers(den)
                den = model.pca_smooth(den, alpha=a)
            cd_pred = ev.chamfer_distance(den, gt, normalize=True)
            cd_s.append(ev.metric_to_score(cd_pred, cd_noisy))
            if mv is not None:
                p2s_pred = ev.point_to_surface_distance(den, mv, mf, normalize_ref_pc=gt)
                p2s_s.append(ev.metric_to_score(p2s_pred, p2s_noisy))
        mcd = float(np.mean(cd_s))
        mp2s = float(np.mean(p2s_s)) if p2s_s else 0.0
        print(f"{a:>6.2f} | {mcd:>8.2f} | {mp2s:>9.2f} | {0.5*mcd+0.5*mp2s:>6.2f}")


if __name__ == '__main__':
    main()
