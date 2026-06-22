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

### 2026-06-23 — local eval harness + first baseline
- Added a local evaluation path (no official GT needed): `datalist/local_test.txt`
  (32 shapes, stratified to the test category mix), `tools/make_local_valset.py`
  (mesh-frame clean GT + noisy), `scripts/eval_local.sh` (denoise + evaluate.py),
  installed `point_cloud_utils` for accurate P2S.
- **Baseline @ checkpoint_7 (epoch 7): final 43.39** (CD 33.78, P2S 53.01).
  CD_pred 1.70e-4 vs CD_noisy 2.63e-4; P2S_pred 7.2e-5 vs P2S_noisy 1.54e-4.
- Target: **>=80**. Gap is large but the model is very early (7/100 epochs) and
  training is slow (~30 min/epoch, num_workers=0). Plan to close the gap:
  1. **Throughput first** (critical path): cache mesh-sampled clouds so epochs are
     fast enough to actually train to convergence; then raise base `num_samples`
     toward 50k to match test density.
  2. Re-eval the local set at later checkpoints to track the trajectory.
  3. Predict-side tuning on a fixed checkpoint (cheap, no retrain): `num_iters` L,
     `predict_seed_k` (patch coverage), noise-variance estimate — sweep for best score.
  4. If quality plateaus: revisit patch_size/kp_mask, lambda, model capacity.

### 2026-06-23 — run #6: cloud cache + 50k density (the throughput fix)
- Pre-sampled 3851 train + 100 val meshes to 50k-point clouds (`dataset_cache`, 2.3G),
  added `npy_cloud` loader; training now np.loads clouds + crops patches.
- **Epoch time 23 min -> ~4.3 min (~5x faster)** at 3.9 it/s, even on 2 GPUs.
  (GPUs 0,1 were taken by another user `xuewenhan`; run #6 uses GPUs 2,3, NP=2.)
  100 epochs now ~7h instead of ~50h. Patch density now matches the 50k test clouds.
- Trimesh load + surface sampling per step was indeed the bottleneck.
- Next: eval at checkpoint ~30 to confirm trajectory toward 80; then predict-side
  sweep (L, seed_k) on the best checkpoint.

### 2026-06-23 — run #6 eval@15 + predict sweep findings
- **Local eval @ checkpoint_15 (run #6): 47.88** (CD 36.95, P2S 58.82). Up from
  43.39 but slow; **P2S >> CD** => points reach the surface but CD (distribution)
  lags.
- **noise_scale sweep (tools/sweep_predict.py) is FLAT**: scale 1->8 moves tau
  40->114 but final score stays ~39 on 6 tables (CD ~28.6, P2S ~49.3). =>
  NOT under-denoising; denoising saturates. Bottleneck = model score quality.
  Keep `predict_noise_scale: 1.0`.
- Interpretation: epoch 15 is early (cosine LR ~94% of peak); score denoisers
  gain most as LR anneals (epochs ~60-100). Plan: keep training, measure the
  slope at checkpoint_40, then 80. If it plateaus < 80, raise capacity
  (feat_dim 128->256, grad_hidden) and/or add light output smoothing/outlier
  removal (vm-style post-proc improved CD/P2S there).

### 2026-06-23 — root-cause diagnosis + capacity pivot (run #7)
- **Diagnostics (tools/diagnose.py)** ruled out a structural bug: CD splits into
  pred->GT (surface, improves ~45-60%) and GT->pred (coverage, already small &
  stable); denoised NN-spacing (0.0088) > GT spacing (0.0056) => NOT clustering.
  Mechanism correct; the model just leaves a rough surface (= undertrained / low capacity).
- **Leakage quantified**: full-32 47.88 vs true-holdout-19 **45.45** (~2.4 pt
  inflation). Added `datalist/local_holdout.txt` + `evaluate.py --keys_list` for
  honest holdout scoring.
- **Post-proc disabled** (`predict_post_smooth: false`): it helps a weak model
  (+7.5) but is a crutch that hurts a strong one (per user). Focus on the model.
- **Capacity pivot (run #7)**: SDN used feat_dim=128 vs the repo's vm baseline
  (76 pts) at 512 -> raised feat_dim/grad_hidden 128->256; training set now
  EXCLUDES all eval shapes (0 overlap, true holdout). ~10 min/epoch on GPUs 2,3.
- Plan: train to convergence; eval true holdout at checkpoints 15/30/60. If
  feat256 underperforms, go 512. Other levers: lambda tuned to 0.5-2% noise,
  denser NN reference for GT score, longer training.

### 2026-06-23 — CRITICAL: loss-weight collapse bug found & fixed (run #8)
- feat256 holdout **DECREASED** with training: e15=39.09 -> e30=33.86, while train
  loss kept dropping. Loss is MISALIGNED with the metric. (feat256@15 < feat128@15
  because the bigger model fits the broken objective faster.)
- **Root cause**: the earlier `weight_sigma_floor = sigma_bar(t_min) = 3e-5` (added
  to stop 1e7 loss spikes) makes the weight `(1-λ)/σ̄+λ` ≈ **317** for tiny-noise and
  the stage-2 s=0 (clean target, displacement≈0) cases. The loss is dominated by
  "predict ≈ zero", so the net collapses toward NO displacement -> denoises LESS as
  it trains. This explains rough surfaces, flat noise_scale sweep, and the
  decreasing trajectory all at once.
- **Fix**: `weight_sigma_floor: 0.005` (competition noise scale) -> weight capped at
  ~3, balanced across noise levels. Reverted feat_dim 256->128 for fast validation
  (capacity wasn't the issue; the bigger model just collapsed faster).
- run #8 = feat128 + floor 0.005 + clean holdout + post-proc off. Success criterion:
  holdout score now INCREASES with training (vs 39->34). Eval at ckpt 15/30.
