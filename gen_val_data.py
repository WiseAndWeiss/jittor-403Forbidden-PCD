"""Generate a local validate-set evaluation dataset (clean + noisy point clouds)
from validate-split meshes, so we can compute real CD/P2S scores locally without
the competition platform. Noise std 0.015 matches the test set's P2S_noisy.
"""
import os, sys
import numpy as np
import trimesh

ROOT = os.path.expanduser('~/starter_code')
N_PTS = 50000
NOISE_STD = 0.015
SEED = 1234
K = int(sys.argv[1]) if len(sys.argv) > 1 else 20

ids = [l.strip() for l in open(os.path.join(ROOT, 'datalist/validate.txt')) if l.strip()][:K]
out_root = os.path.join(ROOT, 'val_eval')
os.makedirs(out_root, exist_ok=True)

rng = np.random.RandomState(SEED)
ok = 0
for i, vid in enumerate(ids):
    mesh_path = os.path.join(ROOT, 'dataset_train', vid, 'models/model_normalized.obj')
    if not os.path.exists(mesh_path):
        print('MISS mesh', vid); continue
    m = trimesh.load(mesh_path, process=False, force='mesh')
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    pts, _ = trimesh.sample.sample_surface(m, N_PTS)
    pts = np.asarray(pts, dtype=np.float64)
    # Keep points in the mesh's NATIVE coordinate frame (do NOT normalize),
    # otherwise clean/pred would be in a different frame than the mesh and P2S
    # (which uses the mesh) breaks. evaluate.py normalizes by gt internally.
    c = (pts.max(0) + pts.min(0)) / 2.0
    s = np.sqrt(((pts - c) ** 2).sum(1)).max()   # bounding radius
    # noise std proportional to scale so the normalized perpendicular noise ~0.015
    noisy = pts + rng.normal(0, NOISE_STD * s, pts.shape)
    d = os.path.join(out_root, vid)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, 'clean.npy'), pts.astype(np.float32))
    np.save(os.path.join(d, 'noisy.npy'), noisy.astype(np.float32))
    ok += 1
    print('%2d/%d %s' % (i + 1, len(ids), vid))

# write a datalist for predict
with open(os.path.join(ROOT, 'datalist/validate_eval.txt'), 'w') as f:
    for vid in ids:
        if os.path.exists(os.path.join(out_root, vid, 'noisy.npy')):
            f.write(vid + '\n')
print('done: %d/%d clouds -> %s' % (ok, len(ids), out_root))
