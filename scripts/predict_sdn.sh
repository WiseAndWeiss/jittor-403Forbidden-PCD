#!/usr/bin/env bash
# =============================================================================
# One-click inference + submission packaging for the SDN denoiser.
#
#   bash scripts/predict_sdn.sh                       # uses load_ckpt from yaml
#   CKPT=experiments/sdn/checkpoint_99.pkl bash scripts/predict_sdn.sh
#
# Produces result/shapenet/<synset>/<model>/denoised.npy and packs result_sdn.zip
# Inference is single-GPU (the adaptive schedule is sequential per shape).
# =============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jittor

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TASK="${TASK:-configs/task/predict_sdn.yaml}"

# Optional checkpoint override: rewrite load_ckpt in a temp task copy.
if [ -n "${CKPT:-}" ]; then
  TMP="$(mktemp --suffix=.yaml)"
  sed "s#^load_ckpt:.*#load_ckpt: ${CKPT}#" "$TASK" > "$TMP"
  TASK="$TMP"
  echo "[predict_sdn] overriding checkpoint -> $CKPT"
fi

echo "[predict_sdn] task=$TASK gpu=$CUDA_VISIBLE_DEVICES"
python run.py --task "$TASK"

# ---- package submission (find the produced shapenet/ dir, zip its parent) ----
SHAPENET_DIR="$(find result -type d -name shapenet 2>/dev/null | head -1)"
if [ -n "$SHAPENET_DIR" ]; then
  OUT="$REPO/result_sdn.zip"
  rm -f "$OUT"
  ( cd "$(dirname "$SHAPENET_DIR")" && zip -rq "$OUT" shapenet/ )
  echo "[predict_sdn] wrote $OUT"
else
  echo "[predict_sdn] WARNING: no result/shapenet directory found"
fi
