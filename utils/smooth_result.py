#!/usr/bin/env python3
"""
Post-process denoised point clouds with bilateral smoothing.

Usage:
    # Smooth all results and save to a new directory
    python utils/smooth_result.py --input_dir result --output_dir result_smooth

    # Smooth a single file
    python utils/smooth_result.py --input_file result/shapenet/02691156/xxx/denoised.npy --output_dir result_smooth

    # Custom smoothing parameters
    python utils/smooth_result.py --input_dir result --output_dir result_smooth --sigma_s 0.02 --sigma_c 0.4 --k 16

Parameters:
    --sigma_s: Spatial sigma (default 0.03). Larger = more spatial smoothing.
               In unit-sphere coordinates (radius ~1.0), 0.03 ≈ 2× avg nearest-neighbor distance for 32K pts.
    --sigma_c: Normal consistency sigma (default 0.3). Smaller = stricter edge preservation.
               When < 0.1, only points with nearly identical normals influence each other (strict edge).
               When > 0.5, normals can differ significantly, leading to more aggressive smoothing.
    --k:        Number of neighbors (default 24).
                Higher = more neighbors in local PCA → more stable normal estimates,
                but may cross sharp edges. 24 is 2× the minimum for stable 3×3 covariance.
"""

import os
import sys
import argparse
import numpy as np
from glob import glob
from scipy.spatial import cKDTree
from tqdm import tqdm


def estimate_normals(points, k=24):
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)
    neighbors = points[idx[:, 1:]]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', centered, centered) / k
    _, eigenvectors = np.linalg.eigh(cov)
    return eigenvectors[:, :, 0]


def bilateral_smooth(points, sigma_s=0.03, sigma_c=0.3, k=24):
    normals = estimate_normals(points, k=k)
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


def find_npy_files(input_dir):
    return glob(os.path.join(input_dir, "**", "denoised.npy"), recursive=True)


def process_file(input_path, output_path, sigma_s, sigma_c, k):
    points = np.load(input_path).astype(np.float64)
    smoothed = bilateral_smooth(points, sigma_s=sigma_s, sigma_c=sigma_c, k=k)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, smoothed.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="Bilateral smoothing post-process for denoised point clouds")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing result shapenet/.../denoised.npy files")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Single denoised.npy file to process")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (maintains same shapenet/.../denoised.npy structure)")
    parser.add_argument("--sigma_s", type=float, default=0.03,
                        help="Spatial sigma (default: 0.03)")
    parser.add_argument("--sigma_c", type=float, default=0.3,
                        help="Normal consistency sigma (default: 0.3)")
    parser.add_argument("--k", type=int, default=24,
                        help="Number of neighbors (default: 24)")
    args = parser.parse_args()

    if args.input_dir is None and args.input_file is None:
        print("Error: Must specify either --input_dir or --input_file")
        sys.exit(1)

    if args.input_file:
        files = [args.input_file]
        base_dir = os.path.dirname(args.input_file)
        while os.path.basename(base_dir) not in ("", "result"):
            base_dir = os.path.dirname(base_dir)
        base_dir = os.path.dirname(base_dir) if base_dir else "."
    else:
        files = find_npy_files(args.input_dir)
        base_dir = os.path.abspath(args.input_dir)

    print(f"Found {len(files)} files to process")
    print(f"Parameters: sigma_s={args.sigma_s}, sigma_c={args.sigma_c}, k={args.k}")

    for f in tqdm(files):
        rel_path = os.path.relpath(f, base_dir)
        output_path = os.path.join(args.output_dir, rel_path)
        process_file(f, output_path, args.sigma_s, args.sigma_c, args.k)

    print(f"Done. Smoothed results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
