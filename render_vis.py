import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ROOT = os.path.expanduser('~/starter_code')
PRED = 'val_pred_platmatch'   # b2 (final model) predictions
ids = [l.strip() for l in open(f'{ROOT}/datalist/platmatch.txt') if l.strip()]

# pick one representative shape per category (table / airplane / sofa)
WANT = [('04379243', 'Table'), ('02691156', 'Airplane'), ('04256520', 'Sofa')]
picks = []
for cat, name in WANT:
    for vid in ids:
        if vid.split('/')[1] == cat:
            picks.append((vid, name)); break

N_SHOW = 8000
VMAX = 0.03                    # 误差着色上限(归一化坐标)
rng = np.random.RandomState(0)

def norm(p, c, s):
    return (p - c) / s

def err_to_clean(p, clean):
    return cKDTree(clean).query(p)[0]

fig = plt.figure(figsize=(11, 3.6 * len(picks)))
col_titles = ['Noisy', 'Ours (denoised)', 'Clean (GT)']

for r, (vid, name) in enumerate(picks):
    clean = np.load(f'{ROOT}/val_platmatch/{vid}/clean.npy').astype(np.float64)
    noisy = np.load(f'{ROOT}/val_platmatch/{vid}/noisy.npy').astype(np.float64)
    pred  = np.load(f'{ROOT}/{PRED}/{vid}/denoised.npy').astype(np.float64)
    c = (clean.max(0) + clean.min(0)) / 2.0
    s = np.sqrt(((clean - c) ** 2).sum(1)).max()
    cl, no, pr = norm(clean, c, s), norm(noisy, c, s), norm(pred, c, s)

    # 误差(到真值最近点距离)
    e_no = err_to_clean(no, cl)
    e_pr = err_to_clean(pr, cl)
    e_cl = np.zeros(len(cl))

    # 子采样
    def sub(p, e):
        idx = rng.choice(len(p), min(N_SHOW, len(p)), replace=False)
        return p[idx], e[idx]
    panels = [sub(no, e_no), sub(pr, e_pr), sub(cl, e_cl)]

    # 统一视角:用真值的主轴方向
    for k, (pts, e) in enumerate(panels):
        ax = fig.add_subplot(len(picks), 3, r * 3 + k + 1, projection='3d')
        sc = ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=e, cmap='jet',
                        vmin=0, vmax=VMAX, s=1.2, linewidths=0)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        if r == 0:
            ax.set_title(col_titles[k], fontsize=12)
        if k == 0:
            ax.text2D(-0.08, 0.5, name, transform=ax.transAxes,
                      rotation=90, va='center', fontsize=12)

cbar_ax = fig.add_axes([0.92, 0.25, 0.013, 0.5])
fig.colorbar(sc, cax=cbar_ax, label='dist. to GT surface (normalized)')
plt.subplots_adjust(left=0.04, right=0.90, top=0.95, bottom=0.02, wspace=0.02, hspace=0.05)
plt.savefig(f'{ROOT}/report_vis.png', dpi=150, bbox_inches='tight')
print('saved report_vis.png')
for vid, name in picks:
    print(' ', name, vid)
