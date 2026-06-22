#!/usr/bin/env python3
"""Build a local validation/test set carved out of the training meshes, in a
form that `evaluate.py` can score (CD + P2S).

For each shape in a datalist we:
  1. sample `num_points` points uniformly on the mesh surface  -> clean GT
  2. add isotropic Gaussian noise, std = u * R  (R = bounding-sphere radius,
     u ~ U[noise_min, noise_max]) to mimic the competition's 0.5%-2% noise
     "normalized to the unit sphere"                            -> noisy input

Everything is written in the *mesh coordinate frame* so that `evaluate.py`'s
point-to-surface metric (which normalises the mesh by the GT's bounding box) is
self-consistent. The competition score is a ratio (1 - pred/noisy), hence
scale-invariant, so this stays comparable to the official leaderboard.

Output layout (keys mirror the datalist, e.g. shapenet/<synset>/<model>):
    <out>/noisy/<key>/noisy.npy     (N,3) float32   -- model input
    <out>/gt/<key>/clean.npy        (N,3) float32   -- ground-truth clean

Meshes for P2S are read directly from `--mesh_dir` (the training set), so no
copies are made.

Usage:
    python tools/make_local_valset.py \
        --list datalist/local_test.txt --mesh_dir dataset_train \
        --out local_eval --num_points 50000 --seed 2026
"""
import argparse
import os

import numpy as np
import trimesh


def bounding_sphere_radius(pc):
    c = (pc.max(0) + pc.min(0)) / 2.0
    return float(np.sqrt(((pc - c) ** 2).sum(1)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', default='datalist/local_test.txt')
    ap.add_argument('--mesh_dir', default='dataset_train')
    ap.add_argument('--mesh_name', default='models/model_normalized.obj')
    ap.add_argument('--out', default='local_eval')
    ap.add_argument('--num_points', type=int, default=50000)
    ap.add_argument('--noise_min', type=float, default=0.005)  # relative to unit sphere
    ap.add_argument('--noise_max', type=float, default=0.020)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    keys = [l.strip() for l in open(args.list) if l.strip()]
    rng = np.random.RandomState(args.seed)
    print(f"[make_local_valset] {len(keys)} shapes -> {args.out}/  "
          f"(num_points={args.num_points}, noise u~U[{args.noise_min},{args.noise_max}])")

    ok = 0
    for i, key in enumerate(keys):
        mesh_path = os.path.join(args.mesh_dir, key, args.mesh_name)
        if not os.path.exists(mesh_path):
            print(f"  [skip] missing mesh: {mesh_path}")
            continue
        mesh = trimesh.load(mesh_path, process=False, force='mesh')
        if mesh.vertices is None or len(mesh.faces) == 0:
            print(f"  [skip] empty mesh: {key}")
            continue

        # 1) clean GT: uniform surface sampling
        clean, _ = trimesh.sample.sample_surface(mesh, args.num_points)
        clean = np.asarray(clean, dtype=np.float32)

        # 2) noisy input: noise scaled to the bounding-sphere radius
        R = bounding_sphere_radius(clean)
        u = rng.uniform(args.noise_min, args.noise_max)
        noise = rng.randn(*clean.shape).astype(np.float32) * (u * R)
        noisy = (clean + noise).astype(np.float32)

        noisy_dir = os.path.join(args.out, 'noisy', key)
        gt_dir = os.path.join(args.out, 'gt', key)
        os.makedirs(noisy_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        np.save(os.path.join(noisy_dir, 'noisy.npy'), noisy)
        np.save(os.path.join(gt_dir, 'clean.npy'), clean)
        ok += 1
        print(f"  [{i+1}/{len(keys)}] {key}  R={R:.3f} noise={u*100:.2f}%xR")

    print(f"[make_local_valset] done: {ok}/{len(keys)} written under {args.out}/")


if __name__ == '__main__':
    main()
