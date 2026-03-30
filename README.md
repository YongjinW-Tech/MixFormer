# MixFormer 公开数据集复现

复现论文《MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders》的核心方法设计

当前实现重点对齐论文中的主干结构：

- Feature Embedding and Splitting
- Query Mixer
- Cross Attention
- Output Fusion
- UI-MixFormer 风格的 user-item decoupling 开关

同时，代码针对公开数据集做了可运行化适配，包括通用下载、预处理、训练、评估和效率指标统计。

## 1. 当前支持

- 支持数据集：`MovieLens-100K`、`MovieLens-1M`、`Amazon All Beauty`、`Amazon Video Games`、`Amazon Electronics x1`、`MIND-small`、`TaobaoAd x1`
- 训练入口：`scripts/train.py`
- 通用下载入口：`scripts/download_dataset.py`
- 通用预处理入口：`scripts/preprocess_dataset.py`
- 兼容保留：`scripts/download_ml1m.py`、`scripts/preprocess_ml1m.py`
- 训练输出：最佳模型 `best_model.pt`，指标文件 `metrics.json`

## 2. 环境准备

建议在仓库根目录创建虚拟环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

主要依赖如下：

- `torch`
- `PyYAML`
- `tqdm`
- `pandas`
- `numpy`

## 3. 数据集与配置文件

| 数据集 | 下载参数 | 默认原始目录 | 推荐训练配置 |
| --- | --- | --- | --- |
| MovieLens-100K | `ml-100k` | `data/raw/ml-100k` | `configs/ml100k_mixformer.yaml` |
| MovieLens-1M | `ml-1m` | `data/raw/ml-1m` | `configs/ml1m_mixformer.yaml`、`configs/ml1m_ui_mixformer.yaml` |
| Amazon All Beauty | `amazon-all-beauty` | `data/raw/amazon-all-beauty` | `configs/amazon_all_beauty_mixformer.yaml`、`configs/amazon_all_beauty_mixformer_96g.yaml` |
| Amazon Video Games | `amazon-video-games` | `data/raw/amazon-video-games` | `configs/amazon_video_games_mixformer.yaml`、`configs/amazon_video_games_mixformer_96g.yaml` |
| Amazon Electronics x1 | `amazon-electronics-x1` | `data/raw/amazon-electronics-x1` | `configs/amazon_electronics_x1_mixformer.yaml`、`configs/amazon_electronics_x1_mixformer_96g.yaml` |
| MIND-small | `mind-small` | `data/raw/mind-small` | `configs/mind_small_mixformer.yaml`、`configs/mind_small_mixformer_96g.yaml` |
| TaobaoAd x1 | `taobao-ad-x1` | `data/raw/taobao-ad-x1` | `configs/taobao_ad_x1_mixformer_96g.yaml` |

说明：

- 文件名带 `_96g` 的配置，当前代码里主要体现为更大的 `batch_size` 和 `eval_batch_size`。
- `MovieLens-1M` 的训练配置默认读取 `data/processed/ml1m_mixformer.pkl`，这一点和通用预处理脚本的默认输出名不完全一致，下面会单独说明。

## 4. 快速开始

以 `MovieLens-1M` 为例：

```bash
.venv/bin/python scripts/download_dataset.py --dataset ml-1m
.venv/bin/python scripts/preprocess_dataset.py \
  --dataset ml-1m \
  --output-path data/processed/ml1m_mixformer.pkl
.venv/bin/python scripts/train.py --config configs/ml1m_mixformer.yaml
```

如果想快速检查训练链路是否正常，可以先跑一个 smoke test：

```bash
.venv/bin/python scripts/train.py \
  --config configs/ml1m_mixformer.yaml \
  --epochs 1 \
  --device cpu \
  --limit-train 2048 \
  --limit-val 512 \
  --limit-test 512 \
  --eval-negatives 20 \
  --output-dir outputs/ml1m_mixformer_smoke
```

## 5. 数据下载

通用下载脚本支持以下数据集：

```bash
.venv/bin/python scripts/download_dataset.py --dataset ml-100k
.venv/bin/python scripts/download_dataset.py --dataset ml-1m
.venv/bin/python scripts/download_dataset.py --dataset amazon-all-beauty
.venv/bin/python scripts/download_dataset.py --dataset amazon-video-games
.venv/bin/python scripts/download_dataset.py --dataset amazon-electronics-x1
.venv/bin/python scripts/download_dataset.py --dataset mind-small
.venv/bin/python scripts/download_dataset.py --dataset taobao-ad-x1
```

其中：

- `ml-100k`、`ml-1m` 会自动下载并解压。
- `amazon-all-beauty`、`amazon-video-games` 会下载原始 `json.gz` 文件。
- `amazon-electronics-x1`、`mind-small` 会下载并解压对应公开镜像数据。
- `taobao-ad-x1` 会下载 `TaobaoAd_x1.zip` 到 `data/raw/taobao-ad-x1/`，预处理脚本可以直接读取这个 zip。

兼容旧版 `MovieLens-1M` 下载方式：

```bash
.venv/bin/python scripts/download_ml1m.py --output-dir data/raw
```

## 6. 数据预处理

通用预处理命令：

```bash
.venv/bin/python scripts/preprocess_dataset.py --dataset <dataset-name>
```

常用示例：

```bash
.venv/bin/python scripts/preprocess_dataset.py --dataset ml-100k
.venv/bin/python scripts/preprocess_dataset.py \
  --dataset ml-1m \
  --output-path data/processed/ml1m_mixformer.pkl
.venv/bin/python scripts/preprocess_dataset.py --dataset amazon-all-beauty
.venv/bin/python scripts/preprocess_dataset.py --dataset amazon-video-games
.venv/bin/python scripts/preprocess_dataset.py --dataset amazon-electronics-x1
.venv/bin/python scripts/preprocess_dataset.py --dataset mind-small
.venv/bin/python scripts/preprocess_dataset.py --dataset taobao-ad-x1
```

可选参数：

- `--raw-path`：覆盖默认原始数据目录
- `--output-path`：覆盖默认输出路径
- `--min-rating`：评分阈值，默认 `4`
- `--min-user-interactions`：保留用户的最小交互数，默认 `5`

预处理规则概览：

- `MovieLens-100K`、`MovieLens-1M`、Amazon 2018 两个子集会按时间排序，并构造 next-item 风格的 `train / val / test`
- `Amazon Electronics x1`、`MIND-small`、`TaobaoAd x1` 会读取各自公开格式，再映射到当前统一的 MixFormer 输入结构
- 所有预处理结果都会保存成一个 pickle bundle，训练脚本直接读取

兼容旧版 `MovieLens-1M` 预处理方式：

```bash
.venv/bin/python scripts/preprocess_ml1m.py \
  --raw-dir data/raw/ml-1m \
  --output-path data/processed/ml1m_mixformer.pkl
```

## 7. 模型训练

基础训练命令：

```bash
.venv/bin/python scripts/train.py --config <config-path>
```

示例：

```bash
.venv/bin/python scripts/train.py --config configs/ml100k_mixformer.yaml
.venv/bin/python scripts/train.py --config configs/ml1m_mixformer.yaml
.venv/bin/python scripts/train.py --config configs/ml1m_ui_mixformer.yaml
.venv/bin/python scripts/train.py --config configs/amazon_all_beauty_mixformer.yaml
.venv/bin/python scripts/train.py --config configs/amazon_video_games_mixformer.yaml
.venv/bin/python scripts/train.py --config configs/amazon_electronics_x1_mixformer.yaml
.venv/bin/python scripts/train.py --config configs/mind_small_mixformer.yaml
.venv/bin/python scripts/train.py --config configs/taobao_ad_x1_mixformer_96g.yaml
```

训练脚本支持以下常用覆盖参数：

- `--epochs`
- `--batch-size`
- `--eval-batch-size`
- `--eval-negatives`
- `--device`
- `--output-dir`
- `--limit-train`
- `--limit-val`
- `--limit-test`

当前训练脚本会：

- 读取 YAML 配置
- 加载预处理后的 bundle
- 构建 `DataLoader`、`BatchBuilder`、`MixFormerModel`
- 使用 `RMSprop` 优化器训练
- 每轮在验证集上按 `NDCG@top_k` 选择最佳模型
- 最后在测试集上输出最终指标

## 8. 训练输出

每次训练会在 `output_dir` 下生成：

- `best_model.pt`：最佳验证指标对应的模型权重
- `metrics.json`：完整训练历史与测试结果

`metrics.json` 里当前会包含：

- `history`：每轮训练和验证指标
- `test_metrics`：测试集指标
- `efficiency_metrics`：参数量、近似 FLOPs、平均延迟
- `device`：实际训练设备

当前评估指标包括：

- `auc`
- `uauc`
- `hr@10`
- `ndcg@10`
- `sampled_auc`

效率统计包括：

- `params_m`
- `approx_gflops_per_batch`
- `avg_latency_ms`

## 9. 与论文的关系

已经较高程度对齐的部分：

- 非序列特征的 embedding、拼接和按 head 切分
- Query Mixer 中的 HeadMixing 与 per-head SwiGLU-FFN
- 由非序列 query 条件化聚合历史序列的 Cross Attention
- Output Fusion 的 per-head 融合
- UI-MixFormer 风格的 user-item decoupling 开关

为了适配公开数据而做的改写：

- 数据集由论文中的私有工业数据替换为公开数据
- 训练目标改为公开 next-item / 点击风格的二分类打分
- 评估指标使用 `HR@10`、`NDCG@10`、`AUC`、`UAUC`
- 训练系统为单机 PyTorch，而不是工业级混合稀疏训练框架

因此，这个仓库更适合被理解为“公开数据上的方法复现”，而不是工业环境的一比一完整复刻。

## 10. 目录结构

```text
configs/
  *.yaml

scripts/
  download_dataset.py
  download_ml1m.py
  preprocess_dataset.py
  preprocess_ml1m.py
  train.py

src/mixformer/
  data/
    dataset.py
    preprocess.py
  models/
    layers.py
    mixformer.py
  reporting.py
  trainer.py
  utils.py
```

核心文件职责：

- `scripts/train.py`：训练主入口
- `src/mixformer/data/preprocess.py`：多数据集解析与 bundle 构建
- `src/mixformer/data/dataset.py`：负采样、padding、batch 构建
- `src/mixformer/models/layers.py`：MixFormer block 关键层实现
- `src/mixformer/models/mixformer.py`：整体模型结构
- `src/mixformer/trainer.py`：训练与评估循环
- `src/mixformer/reporting.py`：AUC、UAUC、参数量、近似 FLOPs、延迟统计

## 11. 当前已知注意点

- `configs/ml1m_mixformer.yaml` 和 `configs/ml1m_ui_mixformer.yaml` 默认读取 `data/processed/ml1m_mixformer.pkl`
- 如果你使用 `scripts/preprocess_dataset.py --dataset ml-1m` 的默认输出路径，它会生成 `data/processed/ml-1m.pkl`
- 所以训练 `ml-1m` 时，要么显式传 `--output-path data/processed/ml1m_mixformer.pkl`，要么直接使用 `scripts/preprocess_ml1m.py`

如果你准备先验证代码链路是否可运行，建议优先从 `ml-100k` 或 `ml-1m` 的 smoke test 开始。
