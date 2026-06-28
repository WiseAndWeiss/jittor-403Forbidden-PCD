# 计图点云降噪 — ASDN + IterativePFN 方法

第六届计图（Jittor）人工智能挑战赛 · 赛道二（深度学习三维点云降噪）参赛方案，基于 Jittor 实现。

---

## 方法概述

分两个阶段。

1. ASDN 主干（Adaptive Score-style Denoising Network）。逐点图卷积提取特征后回归位移，
   一次前向把噪声点云推回干净表面。代码在 `src/model/asdn/`，配置 `configs/model/asdn.yaml`。

2. IterativePFN 堆叠迭代滤波。在 ASDN 主干上堆叠 `num_iters` 个滤波模块，每个模块只对
   上一模块的输出做小幅修正（`schedule: refine`，模块间输入 detach，防止反复去噪造成
   过平滑），并用 ASDN 预训练权重初始化。损失用 InfoCD（`calc_cd_like_InfoV2`，对
   Chamfer 做密度加权，减少聚簇和空洞）。代码在 `src/model/iterpfn/module.py`
   （`IterativePFNModule`），配置 `configs/model/iterpfn_b2.yaml`。

   - `refine`（默认）：每个模块都以干净点云为目标，模块 2 之后看到的是已去噪并 detach
     的输入，只学小幅修正，效果不低于 ASDN。
   - `progressive`：模块 t 的目标是 `clean + α_t·noise`，`α_t=(T-t)/T`，即 IterativePFN
     原文的逐步降噪轨迹（`configs/model/iterpfn_prog.yaml`）。

   推理按 patch 进行：FPS 取种子，每个种子用 kNN 取 1000 点 patch，逐 patch 去噪后按
   最近种子拼回完整点云。

---

## 环境安装

```bash
# 安装计图
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 -y   # gcc/g++ 版本需 <= 10
conda install -c conda-forge libgomp -y          # 确保 OpenMP runtime 存在

# 安装依赖
python -m pip install -r requirements.txt
```

`requirements.txt`：`jittor numpy trimesh scipy omegaconf point-cloud-utils wandb`
（`point-cloud-utils` 仅本地评测用；`wandb` 仅训练日志用，可在 task 配置里关闭）。

---

## 数据准备

```bash
# 训练数据
tar xzf dataset_train.tar.gz
# 解压后：dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj

# 测试数据
unzip dataset_test_noisy.zip
# 解压后：dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy
```

数据清单位于 `datalist/`：`train.txt`（训练）、`test.txt`（测试集）、
`validate.txt` / `validate_eval.txt`（本地留出验证）。

---

## 训练

训练分两步，IterativePFN 需要先有 ASDN 预训练权重。

### 第 1 步：训练 ASDN 主干

```bash
python run.py --task configs/task/train_asdn.yaml
```

- 数据 `train_asdn`（`dataset_train`，`train.txt`），变换 `asdn`，
  Adam lr=5e-4，约 800 epoch。
- 权重保存在 `experiments/asdn/checkpoint_*.pkl`。

### 第 2 步：训练 IterativePFN

`configs/model/iterpfn_b2.yaml` 中的 `init_ckpt` 指向第 1 步的 ASDN 权重
（默认 `experiments/asdn/checkpoint_281.pkl`），如有需要请改为你的最佳 ASDN 权重：

```bash
python run.py --task configs/task/train_iterpfn_b2.yaml
```

- 数据 `train_iterpfn_b8`（batch 8，num_files 10000），变换 `asdn`，
  Adam lr=1e-4，约 12 epoch。
- 权重保存在 `experiments/iterpfn_b2/checkpoint_*.pkl`（提交使用 `checkpoint_9.pkl`）。

断点续训用 `auto_train.sh`，它会从最新 checkpoint 接着训、崩了自动重启：

```bash
bash auto_train.sh configs/task/train_iterpfn_b2.yaml experiments/iterpfn_b2 12
# 参数：<task_yaml> <ckpt_dir> <target_epochs> [base_ckpt]
```

---

## 推理（生成提交文件）

`configs/task/predict_iterpfn_b2_full.yaml` 里已设好权重路径和测试集
（`load_ckpt: experiments/iterpfn_b2/checkpoint_9.pkl`，`dataset_test_noisy`）。
用自己的权重就改 `load_ckpt`，然后运行：

```bash
python run.py --task configs/task/predict_iterpfn_b2_full.yaml
```

降噪结果保存为：
`results/dataset_test_noisy_iterpfn_b2/shapenet/<synset_id>/<model_id>/denoised.npy`
（`np.float32`，shape `(N, 3)`）。

---

## 打包提交

```bash
cd results/dataset_test_noisy_iterpfn_b2
zip -r ../../result.zip shapenet/
```

提交格式：
```
result.zip
  shapenet/
    <synset_id>/
      <model_id>/
        denoised.npy    # np.float32, shape (N, 3)
```

---

## 本地评测（需 GT，仅组委会持有；本地留出自测见 datalist/validate*.txt）

```bash
python evaluate.py \
    --pred_dir ./results/dataset_test_noisy_iterpfn_b2 \
    --gt_dir   ./test_gt \
    --noisy_dir ./dataset_test_noisy \
    --mesh_dir ./dataset_train \
    --workers 8
```

指标为 Chamfer Distance (CD) 与 Point-to-Surface (P2S)，最终分 = 0.5×CD + 0.5×P2S，
效率作为并列时的区分项。

---

## 关键配置说明（`configs/model/iterpfn_b2.yaml`）

| 字段 | 默认值 | 含义 |
|---|---|---|
| `num_iters` | 3 | 堆叠迭代滤波模块数 |
| `schedule` | `refine` | `refine`（每模块监督到干净）/ `progressive`（逐步降噪轨迹） |
| `loss_type` | `infocd` | 训练损失（InfoCD / mse） |
| `detach_between` | true | 模块间输入 detach，防止过平滑 |
| `feat_d_out` | 32 | 特征维度 |
| `nsample` | 16 | 图卷积 kNN 邻居数 |
| `stride_list` | [4,3,2,1] | 主干下采样 stride |
| `predict_patch_size` | 1000 | 推理 patch 点数 |
| `predict_seed_k` | 5 | patch 重叠系数（越大覆盖越全，CD 越好） |
| `predict_stitch` | `best` | patch 拼接策略（按最近种子） |
| `init_ckpt` | `experiments/asdn/checkpoint_281.pkl` | ASDN warm-start 权重 |

可选变体：`iterpfn`（warm-start 自 `asdn_v2`）、`iterpfn_prog`（progressive 调度）、
`iterpfn_t4`（`num_iters=4`）。

---

## 目录结构

```
run.py                          # 统一入口：python run.py --task <yaml>
evaluate.py                     # 本地评测（CD + P2S）
requirements.txt
auto_train.sh                   # 断点续训脚本
configs/
  model/    iterpfn_b2.yaml  asdn.yaml  ...     # 模型超参（__target__ 指定模型类）
  data/     train_iterpfn_b8.yaml  predict.yaml ...
  transform/asdn.yaml                            # 采样/归一化/加噪/裁 patch
  system/   iterpfn_b2.yaml                       # checkpoint 保存目录
  task/     train_iterpfn_b2.yaml                 # 训练任务
            predict_iterpfn_b2_full.yaml          # 提交推理任务
datalist/                       # 训练/测试/验证清单
src/
  data/                         # 数据集、变换、加噪、patch
  model/
    asdn/                       # ASDN 主干（特征提取 + InfoCD 损失）
    iterpfn/module.py           # IterativePFNModule
    parse.py                    # 模型注册：IterativePFNModule
  system/                       # 训练循环、checkpoint、writer
```

---

## 一键复现

```bash
# 0. 环境 + 数据（见上文）
# 1. 训练 ASDN 主干
python run.py --task configs/task/train_asdn.yaml
# 2. 将 configs/model/iterpfn_b2.yaml 的 init_ckpt 指向上一步最佳 ASDN 权重
# 3. 训练 IterativePFN
python run.py --task configs/task/train_iterpfn_b2.yaml
# 4. 生成提交（默认加载 experiments/iterpfn_b2/checkpoint_9.pkl）
python run.py --task configs/task/predict_iterpfn_b2_full.yaml
# 5. 打包
cd results/dataset_test_noisy_iterpfn_b2 && zip -r ../../result.zip shapenet/
```
