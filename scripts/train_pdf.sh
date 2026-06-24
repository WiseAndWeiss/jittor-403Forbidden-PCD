#!/usr/bin/env bash
# =============================================================================
# One-click multi-GPU training for the PD-Flow denoiser (lite + Sinkhorn-EMD).
#
#   bash scripts/train_pdf.sh                                  # auto-pick GPUs
#   CUDA_VISIBLE_DEVICES=0,2,3 bash scripts/train_pdf.sh       # explicit GPUs
#
# BatchNorm runs with sync=False, so MPI data-parallel works without the
# cross-GPU deadlock. Jittor syncs gradients automatically across ranks.
# =============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jittor

# GPU selection: honor CUDA_VISIBLE_DEVICES if set; else pick the free-est cards.
# The server is shared and other users hold large allocations, so "free" here
# means most-free-memory (not <1500 MiB) -- take the top NEED cards, leave 1.
NEED="${NEED:-3}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  TOTAL=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  MAXUSE=$(( TOTAL - 1 ))
  [ "$NEED" -gt "$MAXUSE" ] && NEED="$MAXUSE"
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | sort -t',' -k2 -n | head -n "$NEED" | awk -F',' '{print $1}' | paste -sd, -)
  export CUDA_VISIBLE_DEVICES="$FREE"
fi
NP="${NP:-$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export use_parallel_op_compiler=0
TASK="${TASK:-configs/task/train_pdf.yaml}"

mkdir -p logs experiments/pdf
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/train_pdf_${TS}.log"
ln -sfn "$(basename "$LOG")" logs/train_pdf_latest.log

echo "[train_pdf] repo=$REPO"
echo "[train_pdf] task=$TASK  np=$NP  gpus=$CUDA_VISIBLE_DEVICES"
echo "[train_pdf] log=$LOG"

if [ "$NP" -gt 1 ]; then
  mpirun --allow-run-as-root -np "$NP" \
      python run.py --task "$TASK" 2>&1 | tee "$LOG"
else
  python run.py --task "$TASK" 2>&1 | tee "$LOG"
fi
