#!/usr/bin/env python3
"""
Transplant DistanceModule weights from V1 checkpoint into V2 checkpoint.

Usage:
    python utils/transplant_distance.py \
        --src experiments/vm_straight_v1/checkpoint_best.pkl \
        --dst experiments/vm_straight/checkpoint_147.pkl \
        --out experiments/vm_straight/hybrid.pkl
"""

import argparse
import jittor as jt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="V1 checkpoint (source of DistanceModule)")
    parser.add_argument("--dst", required=True, help="V2 checkpoint (destination, keeps VelocityModules)")
    parser.add_argument("--out", required=True, help="Output hybrid checkpoint path")
    args = parser.parse_args()

    src = jt.load(args.src)
    dst = jt.load(args.dst)

    distance_keys = [k for k in src.keys() if k.startswith("distance_")]
    print(f"Transplanting {len(distance_keys)} distance keys:")
    for k in sorted(distance_keys):
        print(f"  {k}: {list(src[k].shape)}")
        dst[k] = src[k]

    jt.save(dst, args.out)
    print(f"Saved hybrid checkpoint to {args.out}")


if __name__ == "__main__":
    main()
