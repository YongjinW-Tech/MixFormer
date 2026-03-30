from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np
import torch


def binary_auc(scores: Iterable[float], labels: Iterable[int]) -> float:
    scores = np.asarray(list(scores), dtype=np.float64)
    labels = np.asarray(list(labels), dtype=np.int64)
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    ranks = np.zeros_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[start:end] = avg_rank
        start = end

    pos_ranks = ranks[sorted_labels == 1].sum()
    return float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def user_auc(score_groups: Dict[int, Dict[str, List[float]]]) -> float:
    aucs = []
    for payload in score_groups.values():
        auc = binary_auc(payload["scores"], payload["labels"])
        if auc > 0:
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else 0.0


def count_trainable_params_m(model: torch.nn.Module) -> float:
    return sum(param.numel() for param in model.parameters() if param.requires_grad) / 1e6


def estimate_flops_per_batch_g(model_cfg: Dict, batch: Dict[str, torch.Tensor]) -> float:
    # 这里给出的是与论文 Table 1 类似的“每 batch 近似 FLOPs”，
    # 用解析方式估算主要线性层和 attention 计算，不包含 embedding lookup、norm 和 softmax。
    batch_size = int(batch["user_id"].shape[0])
    seq_len = int(batch["hist_item_ids"].shape[1])
    num_heads = int(model_cfg["num_heads"])
    head_dim = int(model_cfg["head_dim"])
    model_dim = num_heads * head_dim
    num_layers = int(model_cfg["num_layers"])
    ffn_multiplier = float(model_cfg["ffn_multiplier"])
    head_hidden_dim = int(head_dim * ffn_multiplier)
    seq_hidden_dim = int(model_dim * ffn_multiplier)

    non_seq_dims = model_cfg["non_seq_embedding_dims"]
    total_non_seq_dim = sum(non_seq_dims.values())
    enable_ui_decoupling = bool(model_cfg.get("enable_ui_decoupling", False))
    num_user_heads = int(model_cfg.get("num_user_heads", num_heads // 2))
    num_item_heads = num_heads - num_user_heads

    flops = 0.0

    if enable_ui_decoupling:
        user_dim = sum(non_seq_dims[key] for key in ["user_id", "gender", "age", "occupation"])
        item_dim = sum(non_seq_dims[key] for key in ["item_id", "genre", "hour", "weekday"])
        user_slice = user_dim // num_user_heads
        item_slice = item_dim // num_item_heads
        flops += batch_size * num_user_heads * 2.0 * user_slice * head_dim
        flops += batch_size * num_item_heads * 2.0 * item_slice * head_dim
    else:
        split_dim = total_non_seq_dim // num_heads
        flops += batch_size * num_heads * 2.0 * split_dim * head_dim

    seq_input_dim = (
        non_seq_dims["item_id"]
        + model_cfg["seq_embedding_dims"]["rating"]
        + non_seq_dims["hour"]
        + non_seq_dims["weekday"]
        + non_seq_dims["genre"]
        + model_cfg["seq_embedding_dims"]["position"]
    )
    flops += batch_size * seq_len * 2.0 * seq_input_dim * model_dim

    for _ in range(num_layers):
        flops += 6.0 * batch_size * num_heads * head_dim * head_hidden_dim
        flops += 6.0 * batch_size * seq_len * model_dim * seq_hidden_dim
        flops += 4.0 * batch_size * seq_len * num_heads * head_dim * head_dim
        flops += 4.0 * batch_size * num_heads * seq_len * head_dim
        flops += 6.0 * batch_size * num_heads * head_dim * head_hidden_dim

    flops += batch_size * 2.0 * model_dim * (model_dim // 2)
    flops += batch_size * 2.0 * (model_dim // 2)
    return flops / 1e9


@torch.no_grad()
def measure_average_latency_ms(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    warmup: int = 2,
    steps: int = 5,
) -> float:
    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    model.eval()
    for _ in range(warmup):
        _ = model(batch)
        sync()

    timings = []
    for _ in range(steps):
        start = time.perf_counter()
        _ = model(batch)
        sync()
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)

    return float(np.mean(timings)) if timings else 0.0


def build_user_score_groups(raw_samples, scores: np.ndarray) -> Dict[int, Dict[str, List[float]]]:
    user_groups: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"scores": [], "labels": []})
    for row_idx, sample in enumerate(raw_samples):
        user_id = sample["user_id"]
        row_scores = scores[row_idx]
        labels = np.zeros_like(row_scores, dtype=np.int64)
        labels[0] = 1
        user_groups[user_id]["scores"].extend(row_scores.tolist())
        user_groups[user_id]["labels"].extend(labels.tolist())
    return user_groups
