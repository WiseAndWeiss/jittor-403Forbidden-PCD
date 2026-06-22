#!/usr/bin/env bash
# =============================================================================
# One-click 3-GPU training for the score-based diffusion denoiser (SDN).
#
#   bash scripts/train_sdn.sh                         # default 3 GPUs, wandb on
#   NP=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_sdn.sh
#   TASK=configs/task/train_sdn.yaml bash scripts/train_sdn.sh
#
# Jittor synchronises gradients automatically across MPI ranks, so distributed
# data-parallel training is just `mpirun -np <N> python run.py ...`.
# =============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate jittor

# Use 3 of the 4 A6000s by default (leave one card for other users).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
NP="${NP:-3}"

# NOTE: Jittor's dataloader workers segfault under MPI (fork-after-CUDA +
# threaded scipy/numpy), so training uses num_workers=0 (data loads in the main
# process). We therefore leave BLAS/OpenMP multi-threaded so main-process data
# prep (mesh sampling + cKDTree) stays fast. Limit threads per rank to avoid the
# 3 ranks oversubscribing the CPU.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
TASK="${TASK:-configs/task/train_sdn.yaml}"

mkdir -p logs experiments/sdn
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/train_sdn_${TS}.log"
ln -sfn "$(basename "$LOG")" logs/train_sdn_latest.log

echo "[train_sdn] repo=$REPO"
echo "[train_sdn] task=$TASK  np=$NP  gpus=$CUDA_VISIBLE_DEVICES"
echo "[train_sdn] log=$LOG"

if [ "$NP" -gt 1 ]; then
  mpirun --allow-run-as-root -np "$NP" \
      python run.py --task "$TASK" 2>&1 | tee "$LOG"
else
  python run.py --task "$TASK" 2>&1 | tee "$LOG"
fi
