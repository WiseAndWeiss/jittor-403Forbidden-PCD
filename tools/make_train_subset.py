#!/usr/bin/env python3
"""Deterministically build a stratified training subset for cached SDN training.

Patch-based denoising does not need all 15k training meshes (the original paper
trains on 40); a few thousand diverse meshes give plenty of patch variety while
keeping the on-disk cloud cache small and fast to build. We cap each category at
`--cap` shapes so the giant table/airplane classes don't dominate.

Usage:
    python tools/make_train_subset.py --train datalist/train.txt \
        --out datalist/train_sdn.txt --cap 400 --seed 2026
"""
import argparse
import collections
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='datalist/train.txt')
    ap.add_argument('--out', default='datalist/train_sdn.txt')
    ap.add_argument('--cap', type=int, default=400)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    by = collections.defaultdict(list)
    for line in open(args.train):
        line = line.strip()
        if line:
            by[line.split('/')[1]].append(line)

    rng = random.Random(args.seed)
    out = []
    for cat in sorted(by):
        pool = sorted(by[cat])
        rng.shuffle(pool)
        out.extend(pool[:args.cap])
    rng.shuffle(out)

    with open(args.out, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print(f"[make_train_subset] {len(out)} shapes ({len(by)} categories, cap {args.cap}) -> {args.out}")


if __name__ == '__main__':
    main()
