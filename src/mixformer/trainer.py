from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .reporting import binary_auc, build_user_score_groups, user_auc


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    batch_builder,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    negative_ratio: int,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for raw_samples in tqdm(loader, desc="train", leave=False):
        batch = batch_builder.build_train_batch(raw_samples, negative_ratio=negative_ratio)
        labels = batch["labels"]
        batch = move_batch_to_device(batch, device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return {"loss": total_loss / max(total_examples, 1)}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable,
    batch_builder,
    device: torch.device,
    num_negatives: int,
    top_k: int,
) -> Dict[str, float]:
    model.eval()
    hr_sum = 0.0
    ndcg_sum = 0.0
    auc_sum = 0.0
    total = 0
    all_scores = []
    all_labels = []
    user_groups = {}

    for raw_samples in tqdm(loader, desc="eval", leave=False):
        batch, group_size = batch_builder.build_eval_batch(raw_samples, num_negatives=num_negatives)
        batch = move_batch_to_device(batch, device)
        logits = model(batch)
        scores = logits.view(len(raw_samples), group_size)
        score_array = scores.detach().cpu().numpy()
        label_array = np.zeros_like(score_array, dtype=np.int64)
        label_array[:, 0] = 1

        pos_scores = scores[:, :1]
        neg_scores = scores[:, 1:]
        rank = 1 + (neg_scores >= pos_scores).sum(dim=1)

        hr_sum += (rank <= top_k).float().sum().item()
        ndcg_sum += ((rank <= top_k).float() / torch.log2(rank.float() + 1.0)).sum().item()
        auc_sum += (pos_scores > neg_scores).float().mean(dim=1).sum().item()
        total += len(raw_samples)

        all_scores.append(score_array.reshape(-1))
        all_labels.append(label_array.reshape(-1))
        batch_user_groups = build_user_score_groups(raw_samples, score_array)
        for user_id, payload in batch_user_groups.items():
            if user_id not in user_groups:
                user_groups[user_id] = {"scores": [], "labels": []}
            user_groups[user_id]["scores"].extend(payload["scores"])
            user_groups[user_id]["labels"].extend(payload["labels"])

    total = max(total, 1)
    flat_scores = np.concatenate(all_scores) if all_scores else np.array([])
    flat_labels = np.concatenate(all_labels) if all_labels else np.array([])
    auc = binary_auc(flat_scores, flat_labels) if len(flat_scores) else 0.0
    uauc = user_auc(user_groups)
    return {
        "auc": auc,
        "uauc": uauc,
        f"hr@{top_k}": hr_sum / total,
        f"ndcg@{top_k}": ndcg_sum / total,
        "sampled_auc": auc_sum / total,
    }
