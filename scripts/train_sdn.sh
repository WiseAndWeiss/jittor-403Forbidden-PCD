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

# GPU selection: honor CUDA_VISIBLE_DEVICES if set; otherwise auto-pick the
# free-est cards (shared server -> avoid colliding with other users). NEED is how
# many we want (default 3, capped to leave 1 card free for others).
NEED="${NEED:-3}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  TOTAL=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  MAXUSE=$(( TOTAL - 1 ))                       # always leave >=1 card for others
  [ "$NEED" -gt "$MAXUSE" ] && NEED="$MAXUSE"
  # cards with <1500 MiB used are considered free
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F',' '$2+0 < 1500 {print $1}' | head -n "$NEED" | paste -sd, -)
  if [ -z "$FREE" ]; then echo "[train_sdn] no free GPU found"; exit 1; fi
  export CUDA_VISIBLE_DEVICES="$FREE"
fi
NP="${NP:-$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')}"

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
