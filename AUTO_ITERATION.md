# SDN Automated Iteration Log

This file is the running log of the automatic train → analyze → retune loop for
the score-based diffusion denoiser (`src/model/sdn/`). A Claude `/loop` polls
training status over SSH and, when a run finishes (or stalls), inspects the loss
/ validation trend, adjusts hyper-parameters or code, and relaunches.

## Loop protocol
1. `ssh … python tools/sdn_status.py` → JSON `{running, max_epoch, recent_epoch_losses, last_error_lines, tail}`.
2. If `running:true` → reschedule a long wakeup (training is hours-long).
3. If finished/stalled:
   - Read `train/*` and `val/*` curves (WandB + log tail).
   - Diagnose: divergence (NaN/inf), under-fitting (flat high loss), over-smoothing,
     instability (loss variance), throughput (GPU util), inference time.
   - Adjust one or two knobs (below), commit to `score-denoising`, redeploy, relaunch.
   - Append a dated entry here.

## Tunable knobs
- `configs/task/train_sdn.yaml`: `lr`, `epochs`, `warmup_epochs`, `grad_clip`, wandb.
- `configs/data/train_sdn.yaml`: `batch_size` (shapes/step), `num_files` (steps/epoch).
- `configs/transform/sdn.yaml`: `num_patches`, `patch_size`, `kp_mask`.
- `configs/model/sdn.yaml`: `lambda_weight` (λ), `t_min`, `num_iters` (L),
  `feat_dim`, `k_grad`, `predict_seed_k`, `predict_patch_step`, `beta_T`.

## Evaluation signals
- No GT for the test set on our side → optimize **validation score-matching loss**
  (`val/<cls>_loss_sum`) and training stability.
- Periodically run `scripts/predict_sdn.sh` on a couple of shapes and eyeball the
  noise-variance estimate τ̂ / output extent for sanity.
- Efficiency matters for ranking (CD/P2S ties broken by speed): watch per-epoch
  time, GPU util, and inference time per shape.

## Log

### 2026-06-22 — initial bring-up
- Implemented SDN, deployed to server, OpenMPI 4.1.6 installed for 3-GPU mpirun.
- Smoke (synthetic) + real-data integration (3 steps, 6 meshes) pass; checkpoint saved.
- Observed: high/noisy early loss from the `(1-λ)/σ̄_t` weighting in stage-2 sampling
  (masked points stay noisy at init while weighted by the small target σ̄_s). Added
  `grad_clip: 5.0` to bound steps. Watch whether loss settles within the first epochs;
  if it stays unstable, candidates: raise `t_min`, lower `lr`, or lower `grad_clip`.
- Next: launch full 3-GPU run (100 epochs, wandb online), then poll & retune.

### 2026-06-22 — run #1 launched (3×A6000, GPUs 0,1,2)
- Bring-up issues fixed: (1) CRLF in deployed `.sh` (→ `.gitattributes` eol=lf);
  (2) my `pkill -f` patterns self-matched the launching shell (→ launch without
  self-matching patterns; use `[p]ython …` bracket trick for the completion waiter);
  (3) **Jittor dataloader workers segfault under MPI** (fork-after-CUDA, si_status 11)
  even though RAM/shm are ample → set `num_workers: 0`.
- Status: training stable. Epoch 0 ~12 min (≈1.8 it/s, 1250 steps), losses healthy
  and decreasing (~0.13 → 0.005 across ranks). WandB run `sdn-pcdenoise/74cojdqi`.
- **Throughput is data-bound** (num_workers=0 → GPU util low/uneven). Top retune
  target for next iteration: restore parallel loading without the MPI worker
  segfault — options: smaller `num_samples` in the `sample` augment to cut per-step
  CPU; try `num_workers: 2`; or a CPU-only worker path. ~12 min/epoch ⇒ ~20 h for
  100 epochs as-is.
- Auto-iteration: a background waiter blocks until the 3 training procs exit
  (finish or crash) and wakes Claude to analyze + retune + relaunch.
