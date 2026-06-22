#!/usr/bin/env python3
"""Diagnose WHY the SDN score is low: break Chamfer into its two directions and
measure point clustering, to tell apart "points not reaching the surface"
(denoising too weak) from "points cluster / leave gaps" (coverage problem).

    python tools/diagnose.py --ckpt experiments/sdn/checkpoint_20.pkl --n 4
"""
import argparse
import os
import sys

import numpy as np
import jittor as jt
jt.flags.use_cuda = 1
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf
from src.model.parse import get_model
import evaluate as ev


def cd_dirs(a, b):
    """return (a->b mean sq dist, b->a mean sq dist) after normalizing by b."""
    b, c, s = ev.normalize_to_unit_sphere(b)
    a = (a - c) / s
    d_ab, _ = cKDTree(b).query(a, k=1)
    d_ba, _ = cKDTree(a).query(b, k=1)
    return (d_ab ** 2).mean(), (d_ba ** 2).mean()


def nn_spacing(p):
    """mean nearest-neighbour distance within a cloud (clustering indicator)."""
    d, _ = cKDTree(p).query(p, k=2)
    return d[:, 1].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--list', default='datalist/local_test.txt')
    ap.add_argument('--eval_dir', default='local_eval')
    ap.add_argument('--n', type=int, default=4)
    args = ap.parse_args()

    mcfg = OmegaConf.to_container(OmegaConf.load('configs/model/sdn.yaml'), resolve=True)
    tcfg = OmegaConf.to_container(OmegaConf.load('configs/transform/sdn.yaml'), resolve=True)
    model = get_model(model_config=mcfg, transform_config=tcfg)
    model.load(args.ckpt); model.set_predict(True); model.eval()

    keys = [l.strip() for l in open(args.list) if l.strip()][:args.n]
    print(f"{'key':>16} | {'cd_n_a2b':>9} {'cd_n_b2a':>9} | {'cd_d_a2b':>9} {'cd_d_b2a':>9} "
          f"| {'sp_gt':>7} {'sp_noisy':>8} {'sp_den':>7}")
    for k in keys:
        noisy = np.load(os.path.join(args.eval_dir, 'noisy', k, 'noisy.npy')).astype(np.float64)
        gt = np.load(os.path.join(args.eval_dir, 'gt', k, 'clean.npy')).astype(np.float64)
        den, tau = model.adaptive_denoise(jt.array(noisy.astype(np.float32)))
        den = den.detach().numpy().astype(np.float64)

        n_ab, n_ba = cd_dirs(noisy, gt)      # noisy: a2b (noisy->gt), b2a (gt->noisy)
        d_ab, d_ba = cd_dirs(den, gt)        # denoised
        # spacings normalized by gt scale
        _, c, s = ev.normalize_to_unit_sphere(gt.copy())
        sp_gt = nn_spacing((gt - c) / s)
        sp_no = nn_spacing((noisy - c) / s)
        sp_de = nn_spacing((den - c) / s)
        short = k.split('/')[-1][:14]
        print(f"{short:>16} | {n_ab:>9.2e} {n_ba:>9.2e} | {d_ab:>9.2e} {d_ba:>9.2e} "
              f"| {sp_gt:>7.4f} {sp_no:>8.4f} {sp_de:>7.4f}  tau={tau}")
    print("\nLegend: cd_*_a2b = pred->GT (surface accuracy), cd_*_b2a = GT->pred "
          "(coverage). sp_* = mean NN spacing (sp_den << sp_gt => clustering).")


if __name__ == '__main__':
    main()
