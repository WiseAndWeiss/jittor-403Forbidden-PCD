# ASDN 使用说明（点云降噪）

ASDN = **Adaptive Stop Denoising Network**（论文：Guo et al. 2025, *You should learn to
stop denoising on point clouds in advance*）。这是基于 Jittor 的移植实现，代码在 `src/model/asdn/`。

核心思想：先用一个分类器（ScaleNet）从含噪点云预测**可恢复因子 ρ**（噪声/干净点云的熵之比），
再用一个 U-Net 去噪器；ρ 决定每个样本在 U-Net 里**走几层就停**（停止层 λ∈{2,3,4}），
对已经干净的点提前停止，避免过度去噪。因此训练分**两个阶段**。

> 运行环境、数据解压见仓库根目录 `README.md`。所有命令都在 conda 的 `jittor` 环境里跑：
> `conda activate jittor`。

---

## 一、整体流程

```text
阶段1 训练分类器 ──► experiments/classify/checkpoint_<e>.pkl
        │  把它填进 configs/model/asdn.yaml 的 classify_ckpt
        ▼
阶段2 训练去噪器 ──► experiments/asdn/checkpoint_<e>.pkl
        │  把它填进 configs/task/predict_asdn.yaml 的 load_ckpt
        ▼
推理 ──► results/dataset_test_noisy/.../denoised.npy ──► 打包 result.zip
```

## 二、阶段 1：训练分类器（ScaleNet）

```bash
python run.py --task configs/task/train_classify.yaml
```
- 权重保存到 `experiments/classify/checkpoint_<epoch>.pkl`（每个 epoch 一个）。
- 训练目标是回归 ρ = 熵(noisy)/熵(clean)，loss 为 MSE。
- 训练完，挑一个权重（一般用最后一个 epoch，如 `checkpoint_199.pkl`）。

## 三、把分类器接进去噪器

编辑 `configs/model/asdn.yaml`，把 `classify_ckpt` 指向阶段 1 的权重：
```yaml
classify_ckpt: experiments/classify/checkpoint_199.pkl
```
- 阶段 2 启动时会打印 `[ASDN] loaded pretrained classifier ...` 确认加载成功。
- 分类器在阶段 2 **保持冻结**（层数选择里的 `ceil()` 天然阻断梯度），不会被更新。
- 若保持 `classify_ckpt: null`，去噪器仍能训练，但停止层由**随机初始化**的分类器决定，
  失去论文的自适应停止意义（仅适合快速跑通调试）。

## 四、阶段 2：训练去噪器

```bash
python run.py --task configs/task/train_asdn.yaml
```
- 权重保存到 `experiments/asdn/checkpoint_<epoch>.pkl`。
- loss 为 InfoCD（对 chamfer 距离做的对比式变体），见 `src/model/asdn/loss.py`。

## 五、推理 + 打包提交

1. 编辑 `configs/task/predict_asdn.yaml` 的 `load_ckpt`，指向你选定的阶段 2 权重，例如：
   ```yaml
   load_ckpt: experiments/asdn/checkpoint_799.pkl
   ```
   ⚠️ 默认值是 `experiments/asdn/checkpoint_best.pkl`，但训练器只按 epoch 存
   `checkpoint_<e>.pkl`，**不会**生成 `_best`，所以这里必须手动改成真实存在的权重。
2. 运行推理：
   ```bash
   python run.py --task configs/task/predict_asdn.yaml
   ```
   结果写到 `results/dataset_test_noisy/shapenet/<synset>/<model>/denoised.npy`
   （float32, shape (N,3)）。
3. 打包：
   ```bash
   cd results/dataset_test_noisy
   zip -r ../../result.zip shapenet/
   ```

> 速度提示：测试点云每个 5 万点、共 199 个，整体推理约几小时（FPS 是顺序循环，已做向量化但
> 仍是瓶颈）。建议用 `nohup ... &` 后台跑，SSH 断开也不影响。

---

## 六、Weights & Biases（wandb）日志

训练会把超参数和指标记录到 wandb。

**首次使用先登录一次**（拿你自己的 API key）：
```bash
wandb login
```

在 `configs/task/train_asdn.yaml` / `train_classify.yaml` 里有 `wandb` 配置块：
```yaml
wandb:
  enabled: true          # false 关闭日志
  project: asdn-pcdenoise
  name: null             # 运行名，null 自动生成
  mode: online           # online 上传 / offline 本地存 / disabled 关闭
```
- **不想上传**（或没法联网/没登录）：设 `mode: offline`，数据存到本地 `wandb/offline-run-*`，
  以后用 `wandb sync wandb/offline-run-xxx` 再上传。
- 即使没登录，`wandb.init` 失败也**不会中断训练**（会打印警告并继续）。

**记录的内容**：
- 超参数（config）：seed、components、model（asdn.yaml 全部字段）、optimizer、loss、trainer、
  data、transform —— 一次性写入 wandb run 的 config。
- 训练指标（按 `log_interval`）：`train/loss`、`train/lr`、各 loss 分量 `train/<name>`。
- 每个 epoch：`train/epoch_loss`；若有验证集：`val/<cls>_<name>`、`val/<cls>_loss_sum`。

权重本身存在 `experiments/`（不上传到 wandb，避免大文件）。如需把某个 checkpoint 作为
wandb artifact 上传，可手动 `wandb artifact put`，或告诉我加进训练循环。

---

## 七、常用配置速查

| 文件 | 关键字段 |
|------|---------|
| `configs/model/asdn.yaml` | `classify_ckpt`（阶段1权重）、`feat_d_out`、`nsample`、`stride_list`、`frame_knn`、`predict_patch_size`；可选加 `predict_seed_k`(默认5)、`predict_seed_k_alpha`(默认10) 调推理显存/速度 |
| `configs/task/train_asdn.yaml` | `optimizer.lr`、`trainer.epochs/warmup_epochs/log_interval`、`wandb`、`load_ckpt`（断点续训） |
| `configs/task/train_classify.yaml` | 同上；`system: classify` |
| `configs/task/predict_asdn.yaml` | `load_ckpt`、`writer.save_dir` |
| `configs/system/classify.yaml` | `ckpt_save_dir: experiments/classify` |
| `configs/system/asdn.yaml` | `ckpt_save_dir: experiments/asdn` |

## 八、本地评测（需 GT，组委会持有）

```bash
python evaluate.py \
    --pred_dir ./results/dataset_test_noisy \
    --gt_dir ./test_gt \
    --noisy_dir ./dataset_test_noisy \
    --mesh_dir ./dataset_train --workers 8
```
