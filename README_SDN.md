# SDN — Adaptive & Iterative Score-Based Point Cloud Denoising (Jittor)

Jittor reproduction of **"Adaptive and Iterative Point Cloud Denoising with
Score-Based Diffusion Model"** (Wang, Li, Xin, Tu, 2025; PDF in `paper/`),
adapted for the Jittor 三维点云降噪 track. Code lives in `src/model/sdn/`.

> Environment & data extraction: see the root `README.md`. All commands run in
> the `jittor` conda env (`conda activate jittor`).

## Paper innovations & where they live

| Innovation (paper section) | Implementation |
|---|---|
| Score-based diffusion with **scaling elimination** — keep the point-cloud mean fixed, `x_t = x_0 + σ̄_t·z` (Sec 3.1, Eq 1–3) | `src/model/sdn/diffusion.py` (`DiffusionSchedule`) |
| **Feature fusion** module `F` — fuse `E(x_t)` & `E(x_ref)` w.r.t. timestep (Sec 4.1, Eq 10) | `network.py` `FeatureFusion` |
| **Gradient prediction + gradient fusion** — per-neighbour gradient & importance weight, softmax fusion over k=32 (Eq 11–13) | `network.py` `GradientPrediction`, `ScoreNetwork.predict_score` |
| **Two-stage sampling** training (Algorithm 1) | `module.py` `SDNModule.get_supervised_loss` |
| Interpolated score-matching **loss** `((1-λ)/σ̄ + λ)·(s−S)`, λ=0.99 (Eq 9) | `module.py` |
| **Adaptive schedule** estimation `σ̄²=Var(‖s‖)` + linear search (Eq 15–16) | `diffusion.py` `find_tau`, `module.py` `_estimate_noise_var` |
| **Iterative denoising** with L=5 adaptive steps (Algorithm 2) | `module.py` `adaptive_denoise` |

Only **clean** point clouds are needed for training; the diffusion noise is
injected inside `training_step`. GT scores use the nearest-clean-neighbour
displacement `S(x)=NN(x,x₀)−x` (Eq 14).

## Train (3× A6000, distributed + WandB)

```bash
wandb login                      # once
bash scripts/train_sdn.sh        # 3 GPUs (CUDA_VISIBLE_DEVICES=0,1,2), wandb on
```

- Distributed data-parallel via `mpirun -np 3` — Jittor auto-syncs gradients and
  shards the dataset; only rank-0 logs to WandB and writes checkpoints.
- WandB logs: `train/loss`, `train/score_loss_t`, `train/score_loss_s`,
  `train/lr`, per-card `gpu/<i>/util` & memory, and per-epoch validation losses.
- Checkpoints: `experiments/sdn/checkpoint_<epoch>.pkl`.
- Knobs: `configs/model/sdn.yaml` (T, β_T, λ, L, k, feat_dim, patch sizes),
  `configs/task/train_sdn.yaml` (lr, epochs, batch, wandb),
  `configs/data/train_sdn.yaml` (batch_size = shapes/step).

Run in the background and keep iterating after disconnect:

```bash
nohup bash scripts/train_sdn.sh > logs/nohup_sdn.log 2>&1 &
python tools/sdn_status.py        # JSON status for the auto-iteration loop
```

## Predict + package submission

Edit `configs/task/predict_sdn.yaml` → `load_ckpt`, then:

```bash
bash scripts/predict_sdn.sh                              # or:
CKPT=experiments/sdn/checkpoint_99.pkl bash scripts/predict_sdn.sh
```

Writes `result/shapenet/<synset>/<model>/denoised.npy` and packs `result_sdn.zip`.

## Automated iteration

Training is launched in the background and a Claude `/loop` polls
`tools/sdn_status.py`; when a run finishes it inspects the loss/validation
trend and adjusts hyper-parameters or code, then relaunches. See
`AUTO_ITERATION.md` for the loop protocol and tuning log.
