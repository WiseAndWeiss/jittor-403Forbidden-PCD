#!/usr/bin/env python3
"""Pre-sample training meshes into fixed point clouds on disk.

The per-step bottleneck during training was loading each .obj with trimesh and
sampling its surface every iteration (with num_workers=0, this runs in the main
process and starves the GPUs). We instead sample each mesh ONCE to `num_points`
points and cache `dataset_cache/<key>/cloud.npy`; training then just np.loads the
cloud and crops random patches -- much faster, and at 50k points the patch
density matches the 50k-point test clouds.

Per-epoch diversity still comes from random patch cropping (+ in-model noise).

Usage:
    python tools/cache_clouds.py --list datalist/train_sdn.txt \
        --mesh_dir dataset_train --out dataset_cache --num_points 50000 --workers 16
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import trimesh


def sample_one(task):
    key, mesh_path, out_path, num_points = task
    if os.path.exists(out_path):
        return (key, 'skip')
    try:
        mesh = trimesh.load(mesh_path, process=False, force='mesh')
        if mesh.vertices is None or len(getattr(mesh, 'faces', [])) == 0:
            return (key, 'empty')
        pts, _ = trimesh.sample.sample_surface(mesh, num_points)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, np.asarray(pts, dtype=np.float32))
        return (key, 'ok')
    except Exception as e:
        return (key, f'err:{e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--mesh_dir', default='dataset_train')
    ap.add_argument('--mesh_name', default='models/model_normalized.obj')
    ap.add_argument('--out', default='dataset_cache')
    ap.add_argument('--num_points', type=int, default=50000)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    keys = [l.strip() for l in open(args.list) if l.strip()]
    tasks = [
        (k, os.path.join(args.mesh_dir, k, args.mesh_name),
         os.path.join(args.out, k, 'cloud.npy'), args.num_points)
        for k in keys
    ]
    print(f"[cache_clouds] {len(tasks)} meshes -> {args.out}/ "
          f"({args.num_points} pts, {args.workers} workers)")

    counts = {}
    with Pool(args.workers) as pool:
        for i, (key, status) in enumerate(pool.imap_unordered(sample_one, tasks)):
            tag = status.split(':')[0]
            counts[tag] = counts.get(tag, 0) + 1
            if (i + 1) % 200 == 0 or i + 1 == len(tasks):
                print(f"  {i+1}/{len(tasks)}  {counts}")
    print(f"[cache_clouds] done: {counts}")


if __name__ == '__main__':
    main()
