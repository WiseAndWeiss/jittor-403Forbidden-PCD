#!/usr/bin/env python3
"""Smoke test for the PD-LTS monotone-flow port. Checks (1) build + param count,
(2) INVERTIBILITY of the monotone flow g(f(x))~x (validates spectral norm +
Banach fixed-point solver), (3) denoise runs finite, (4) training_step + backward.
  python tools/pdlts_smoke.py"""
import os
import sys
import time
import numpy as np

import jittor as jt
jt.flags.use_cuda = 1

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf
from src.model.parse import get_model

mcfg = OmegaConf.to_container(OmegaConf.load('configs/model/pdlts.yaml'), resolve=True)
tcfg = OmegaConf.to_container(OmegaConf.load('configs/transform/pdf.yaml'), resolve=True)
model = get_model(model_config=mcfg, transform_config=tcfg)
print('[smoke] params:', sum(p.numel() for p in model.parameters()))

B, M = 2, 512
clean = np.random.randn(B, M, 3).astype(np.float32)
clean /= np.linalg.norm(clean, axis=2, keepdims=True)
noisy = clean + np.random.randn(B, M, 3).astype(np.float32) * 0.02
pc = jt.array(noisy)

# ---- (2) invertibility of the monotone flow (eval mode: fixed spectral weights) ----
model.eval()
inj = model.feat_extract(pc)
aug = model.augment(pc)
x = jt.concat([pc, aug], dim=-1)               # (B,M,channel)
z = model.f(x, inj)
x_rec = model.g(z, inj)
err = float((x_rec - x).abs().max())
rel = float(((x_rec - x).abs().mean()) / (x.abs().mean() + 1e-9))
print(f'[smoke] invertibility g(f(x)): max_abs_err={err:.2e} rel_mean_err={rel:.2e}')

# single-block invertibility
blk = model.blocks[0]
y = blk(x)
x2 = blk.inverse(y)
print(f'[smoke] one FlowAssembly inv err: {float((x2 - x).abs().max()):.2e}')

# ---- (3) denoise finite ----
den = model.denoise(pc)
print('[smoke] denoise', tuple(den.shape), 'finite', bool(np.isfinite(den.numpy()).all()))

# ---- (4) training_step + backward ----
model.train()
clean_j = jt.array(clean)
opt = jt.optim.Adam(model.parameters(), lr=1e-3)
batch = {'pc_noisy': pc, 'pc_clean': clean_j}
for it in range(3):
    t0 = time.time()
    out = model.training_step(batch)
    opt.step(out['loss'])
    print(f'[smoke] train it{it} loss={out["loss"].item():.6f} ({time.time()-t0:.2f}s)')

# ---- realistic per-step timing (B=8, M=1024 like real training) ----
Bb, Mb = 8, 1024
cl = np.random.randn(Bb, Mb, 3).astype(np.float32); cl /= np.linalg.norm(cl, axis=2, keepdims=True)
nz = cl + np.random.randn(Bb, Mb, 3).astype(np.float32) * 0.02
big = {'pc_noisy': jt.array(nz), 'pc_clean': jt.array(cl)}
for it in range(3):
    t0 = time.time()
    out = model.training_step(big)
    opt.step(out['loss'])
    jt.sync_all()
    print(f'[smoke] BIG(B8,M1024) it{it} loss={out["loss"].item():.5f} ({time.time()-t0:.2f}s)')

print('[smoke] OK')
