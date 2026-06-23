#!/usr/bin/env python3
"""Fast smoke test for the PD-Flow model: training_step + backward and
patch-based inference on tiny synthetic clouds.  python tools/pdf_smoke.py"""
import os
import sys
import numpy as np

import jittor as jt
jt.flags.use_cuda = 1

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf
from src.model.parse import get_model

mcfg = OmegaConf.to_container(OmegaConf.load('configs/model/pdf.yaml'), resolve=True)
tcfg = OmegaConf.to_container(OmegaConf.load('configs/transform/pdf.yaml'), resolve=True)
model = get_model(model_config=mcfg, transform_config=tcfg)
print('[smoke] params:', sum(p.numel() for p in model.parameters()))

B, M = 2, 1024
clean = np.random.randn(B, M, 3).astype(np.float32)
clean /= np.linalg.norm(clean, axis=2, keepdims=True)
noisy = clean + np.random.randn(B, M, 3).astype(np.float32) * 0.02
batch = {'pc_noisy': jt.array(noisy), 'pc_clean': jt.array(clean)}

model.train()
opt = jt.optim.Adam(model.parameters(), lr=1e-3)
for it in range(3):
    out = model.training_step(batch)
    opt.step(out['loss'])
    print(f'[smoke] train it{it} cd={out["loss"].item():.6f}')

model.eval(); model.set_predict(True)
N = 3000
c = np.random.randn(N, 3).astype(np.float32); c /= np.linalg.norm(c, axis=1, keepdims=True)
nz = c + np.random.randn(N, 3).astype(np.float32) * 0.02
res = model.predict_step({'pc_noisy': jt.array(nz[None])})
den = res[0]['pc_denoised']
r_no = np.abs(np.linalg.norm(nz, axis=1) - 1).mean()
r_de = np.abs(np.linalg.norm(den, axis=1) - 1).mean()
print('[smoke] predict', den.shape, 'finite', np.isfinite(den).all())
print(f'[smoke] mean|r-1|: noisy={r_no:.4f} denoised={r_de:.4f}')
print('[smoke] OK')
