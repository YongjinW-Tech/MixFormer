# MixFormer Reproduction on MovieLens-1M

This project reproduces the core `MixFormer` design from the paper `MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders` using the public `MovieLens-1M` dataset.

The original paper uses a private industrial dataset with trillions of interactions, hundreds of features, and multi-task objectives. This reproduction keeps the paper's method-level design as close as possible while adapting the benchmark setup to a public dataset:

- Sequential features: user historical interactions sorted by timestamp.
- Non-sequential features: user attributes, target item attributes, and request-time context.
- Core model blocks: `Feature Embedding and Splitting`, `Query Mixer`, `Cross Attention`, `Output Fusion`.
- Optional paper-inspired variant: `UI-MixFormer` style user-item decoupled query mixing.

## 1. Environment

The requested virtual environment is:

```bash
/home/wyj/workspace/MixFormer/.venv
```

Install dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## 2. Data

This reproduction now supports multiple public datasets:

- `MovieLens-100K`
- `MovieLens-1M`
- `Amazon All Beauty`
- `Amazon Video Games`

Download one dataset with the generic downloader:

```bash
.venv/bin/python scripts/download_dataset.py --dataset ml-100k
.venv/bin/python scripts/download_dataset.py --dataset ml-1m
.venv/bin/python scripts/download_dataset.py --dataset amazon-all-beauty
.venv/bin/python scripts/download_dataset.py --dataset amazon-video-games
```

Or keep using the original `MovieLens-1M` downloader:

```bash
.venv/bin/python scripts/download_ml1m.py --output-dir data/raw
```

Preprocess with the generic script:

```bash
.venv/bin/python scripts/preprocess_dataset.py --dataset ml-100k
.venv/bin/python scripts/preprocess_dataset.py --dataset ml-1m
.venv/bin/python scripts/preprocess_dataset.py --dataset amazon-all-beauty
.venv/bin/python scripts/preprocess_dataset.py --dataset amazon-video-games
```

Or use the original `MovieLens-1M` path:

```bash
.venv/bin/python scripts/preprocess_ml1m.py \
  --raw-dir data/raw/ml-1m \
  --output-path data/processed/ml1m_mixformer.pkl
```

The preprocessing logic follows a leave-one-out style next-item setup:

- keep positive interactions with `rating >= 4`
- sort each user by timestamp
- use all but the last two interactions for training prefixes
- use the penultimate interaction for validation
- use the last interaction for test

## 3. Train MixFormer

MovieLens-100K:

```bash
.venv/bin/python scripts/train.py --config configs/ml100k_mixformer.yaml
```

Base MixFormer:

```bash
.venv/bin/python scripts/train.py --config configs/ml1m_mixformer.yaml
```

Quick smoke test:

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

User-item decoupled variant:

```bash
.venv/bin/python scripts/train.py --config configs/ml1m_ui_mixformer.yaml
```

Amazon All Beauty:

```bash
.venv/bin/python scripts/train.py --config configs/amazon_all_beauty_mixformer.yaml
```

Amazon Video Games:

```bash
.venv/bin/python scripts/train.py --config configs/amazon_video_games_mixformer.yaml
```

## 4. Project Layout

```text
configs/
  amazon_all_beauty_mixformer.yaml
  amazon_video_games_mixformer.yaml
  ml100k_mixformer.yaml
  ml1m_mixformer.yaml
  ml1m_ui_mixformer.yaml
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
  trainer.py
  utils.py
```

## 5. What Matches the Paper

- Transformer-style unified backbone for sequence and dense interaction.
- Non-sequential feature concatenation and head splitting.
- `Query Mixer` with `HeadMixing` and per-head `SwiGLU-FFN`.
- `Cross Attention` that uses mixed high-order non-sequential queries to aggregate user history.
- `Output Fusion` with per-head `SwiGLU-FFN`.
- Optional user-item decoupled query mixing mask.

## 6. What Is Adapted

- Dataset: private Douyin ranking data is replaced with `MovieLens-1M`.
- Objective: public next-item ranking with sampled negatives, instead of industrial CTR / multi-task prediction.
- Metrics: `HR@10`, `NDCG@10`, and sampled `AUC`.
- Optimizer: a single dense optimizer is used for simplicity on a small public benchmark.

## 7. Recommended Reproduction Order

1. Run the base `MixFormer` end to end and verify loss decreases.
2. Confirm validation `NDCG@10` is stable.
3. Run the user-item decoupled config and compare metrics and runtime.
4. Then tune `num_layers`, `max_seq_len`, `head_dim`, and `eval_negatives` for stronger results.
# MixFormer
