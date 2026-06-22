#!/usr/bin/env python3
"""Fast self-contained smoke test for the SDN model (no data pipeline).

Validates the new code paths -- two-stage training loss + backward, and the
adaptive iterative inference -- on tiny synthetic clouds. Run on the server:

    python tools/sdn_smoke.py
"""
import os
import sys
import numpy as np

import jittor as jt
jt.flags.use_cuda = 1

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf
from src.model.parse import get_model

model_cfg = OmegaConf.to_container(OmegaConf.load('configs/model/sdn.yaml'), resolve=True)
trans_cfg = OmegaConf.to_container(OmegaConf.load('configs/transform/sdn.yaml'), resolve=True)
model = get_model(model_config=model_cfg, transform_config=trans_cfg)
print('[smoke] model built:', sum(p.numel() for p in model.parameters()), 'params')

# ---- training_step on a synthetic clean patch batch ----
B, P, M = 2, 2, model.patch_size
mask = np.zeros((B, P, M), dtype=np.float32); mask[..., :model.kp_mask] = 1.0
x0 = np.random.randn(B, P, M, 3).astype(np.float32) * 0.2
batch = {'patch_clean': jt.array(x0), 'mask': jt.array(mask)}

model.train()
optimizer = jt.optim.Adam(model.parameters(), lr=1e-4)
for it in range(2):
    out = model.training_step(batch)
    loss = out['loss']
    optimizer.step(loss)
    print(f'[smoke] train it{it} loss={loss.item():.5f} '
          f't={out["score_loss_t"].item():.5f} s={out["score_loss_s"].item():.5f}')

# ---- predict_step / adaptive_denoise on a synthetic noisy cloud ----
model.eval()
model.set_predict(True)
N = 2000
clean = np.random.randn(N, 3).astype(np.float32)
clean /= np.linalg.norm(clean, axis=1, keepdims=True)            # points on a sphere
noisy = clean + np.random.randn(N, 3).astype(np.float32) * 0.02
pbatch = {'pc_noisy': jt.array(noisy[None, ...])}
res = model.predict_step(pbatch)
den = res[0]['pc_denoised']
print('[smoke] predict out shape', den.shape, 'finite', np.isfinite(den).all())
# sanity: denoised should be no worse, ideally closer, to the unit sphere
r_noisy = np.abs(np.linalg.norm(noisy, axis=1) - 1).mean()
r_den = np.abs(np.linalg.norm(den, axis=1) - 1).mean()
print(f'[smoke] mean |r-1|: noisy={r_noisy:.4f} denoised={r_den:.4f}')
print('[smoke] OK')
