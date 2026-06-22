#!/usr/bin/env bash
# =============================================================================
# Local evaluation: denoise a held-out slice of the training meshes with the SDN
# model and score it with evaluate.py (CD + P2S), to track progress toward the
# ~80-point target without the official test GT.
#
#   bash scripts/eval_local.sh                              # latest checkpoint
#   CKPT=experiments/sdn/checkpoint_40.pkl bash scripts/eval_local.sh
#   GPU=3 bash scripts/eval_local.sh                        # which GPU to use
#
# First run auto-builds the local val set (local_eval/{noisy,gt}). Re-run with
# REBUILD=1 to regenerate it.
# =============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jittor
export CUDA_VISIBLE_DEVICES="${GPU:-3}"          # default: spare card (training uses 0,1,2)

LIST="${LIST:-datalist/local_test.txt}"
MESH_DIR="${MESH_DIR:-dataset_train}"
NUM_POINTS="${NUM_POINTS:-50000}"

# ---- 1. build the local val set (clean GT + noisy) if absent ----
if [ "${REBUILD:-0}" = "1" ] || [ ! -d local_eval/noisy ]; then
  echo "[eval_local] building local val set from $LIST ..."
  python tools/make_local_valset.py --list "$LIST" --mesh_dir "$MESH_DIR" \
      --out local_eval --num_points "$NUM_POINTS"
fi

# ---- 2. pick checkpoint (CKPT, else latest) ----
CKPT="${CKPT:-$(ls -t experiments/sdn/checkpoint_*.pkl 2>/dev/null | head -1)}"
if [ -z "$CKPT" ]; then echo "[eval_local] no checkpoint found in experiments/sdn/"; exit 1; fi
echo "[eval_local] checkpoint: $CKPT"

# ---- 3. denoise the local val set ----
TASK="$(mktemp --suffix=.yaml)"
sed "s#^load_ckpt:.*#load_ckpt: ${CKPT}#" configs/task/predict_local_sdn.yaml > "$TASK"
rm -rf local_eval/pred
python run.py --task "$TASK"

# ---- 4. score with evaluate.py (CD + P2S) ----
echo "[eval_local] scoring (full set) ..."
python evaluate.py \
    --pred_dir local_eval/pred \
    --gt_dir local_eval/gt \
    --noisy_dir local_eval/noisy \
    --mesh_dir "$MESH_DIR" \
    --workers 8

# ---- 4b. honest holdout score (shapes NOT in the training subset) ----
if [ -f datalist/local_holdout.txt ]; then
  echo "[eval_local] scoring (TRUE HOLDOUT, no leakage) ..."
  python evaluate.py \
      --pred_dir local_eval/pred \
      --gt_dir local_eval/gt \
      --noisy_dir local_eval/noisy \
      --mesh_dir "$MESH_DIR" \
      --keys_list datalist/local_holdout.txt \
      --workers 8
fi
