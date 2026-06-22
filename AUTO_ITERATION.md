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

### 2026-06-22 — runs #2–#5: crash triage, loss-spike fix, stable run
- **run #2** crashed at epoch-0 end: validation `batch_size=1 < world_size=3`
  (Jittor asserts `batch_size>=world_size` under MPI). Fix: validate `batch_size: 3`;
  also reordered loop to **save checkpoint before validation** so no epoch is lost.
- **MPI dataloader workers are unusable**: `num_workers` 4 and 8 both segfault
  (`SIGCHLD si_status 11`, fork-after-CUDA + threaded scipy/numpy); single-threaded
  BLAS did NOT help. Settled on `num_workers: 0` (stable). Throughput cost accepted
  for now (~23 min/epoch, GPU util data-bound).
- **Loss spikes to ~1e7** (user-reported via WandB): root cause = the
  `(1-λ)/σ̄+λ` weight diverges when stage-2 samples `Δ=t ⇒ s=0` (clean target,
  σ̄=0); the old `1e-8` clamp gave weight ~1e6. Fix: floor σ̄ in the weight at
  `σ̄(t_min)` (≈3e-5 ⇒ weight ≤ ~317). New knob `weight_sigma_floor` (default auto).
  grad_clip=5 had already kept training stable (spikes were a logging artifact).
- **run #5 (current, stable)**: `num_workers=0` + σ̄ floor. Verified: no crash,
  **max loss seen ≈ 618** (was >1e7), losses 0.005–0.05 typical. WandB live.
- **Inference validated end-to-end** on 1 real test shape (checkpoint_0): correct
  `(50000,3)` float32, finite, bbox preserved, **~31 s/shape** ⇒ ~1.7 h for 199.
- **Next iteration targets (priority order):**
  1. Throughput: precompute/cache per-mesh sampled point clouds to `.npy` and add a
     fast loader, to remove trimesh sampling from the main-process loop (workers are
     a dead end under MPI). Target ≪23 min/epoch.
  2. Density match: training patches use `num_samples=32768`; test clouds are 50k —
     consider raising toward 50000 so patch density matches test.
  3. Once a mid/late checkpoint exists, run `scripts/predict_sdn.sh` on the full
     test set and sanity-check CD/P2S behavior + total inference time.
