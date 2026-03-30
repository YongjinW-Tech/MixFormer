from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class MovieLensMixFormerDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class NegativeSampler:
    def __init__(self, all_item_ids: Sequence[int], user_seen_items: Dict[int, Sequence[int]], seed: int = 42):
        self.all_item_ids = list(all_item_ids)
        self.all_item_id_set = set(self.all_item_ids)
        self.num_items = len(self.all_item_ids)
        self.user_seen_items = {uid: set(items) for uid, items in user_seen_items.items()}
        self.random = random.Random(seed)
        self._warned_caps: set[int] = set()

    def _available_items(self, user_id: int, banned_items: Sequence[int] | None = None) -> List[int]:
        banned = set(banned_items or [])
        banned.update(self.user_seen_items.get(user_id, set()))
        return [item_id for item_id in self.all_item_ids if item_id not in banned]

    def available_count(self, user_id: int, banned_items: Sequence[int] | None = None) -> int:
        banned = set(self.user_seen_items.get(user_id, set()))
        for item_id in banned_items or []:
            if item_id in self.all_item_id_set:
                banned.add(item_id)
        return max(self.num_items - len(banned), 0)

    def sample(
        self,
        user_id: int,
        num_samples: int,
        banned_items: Sequence[int] | None = None,
        rng: random.Random | None = None,
    ) -> List[int]:
        effective_num_samples = min(num_samples, self.available_count(user_id=user_id, banned_items=banned_items))

        if effective_num_samples < num_samples:
            if num_samples not in self._warned_caps:
                warnings.warn(
                    (
                        "Requested more training negatives than available unique items for at least one user. "
                        f"Auto-truncating from {num_samples} to {effective_num_samples}."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_caps.add(num_samples)

        if effective_num_samples == 0:
            return []

        banned = set(self.user_seen_items.get(user_id, set()))
        for item_id in banned_items or []:
            if item_id in self.all_item_id_set:
                banned.add(item_id)

        # Large-candidate datasets like Taobao cannot afford an O(num_items) scan per sample.
        # Use rejection sampling when the banned fraction is tiny, and fall back to exact sampling
        # for smaller candidate pools or near-exhaustive requests.
        if self.num_items <= 50000 or effective_num_samples * 4 >= self.available_count(user_id, banned_items):
            available_items = self._available_items(user_id=user_id, banned_items=banned_items)
            sampler = rng or self.random
            return sampler.sample(available_items, k=effective_num_samples)

        sampler = rng or self.random
        selected_items: List[int] = []
        selected_set = set()
        max_attempts = max(effective_num_samples * 20, 100)
        attempts = 0
        while len(selected_items) < effective_num_samples and attempts < max_attempts:
            candidate_item = self.all_item_ids[sampler.randrange(self.num_items)]
            attempts += 1
            if candidate_item in banned or candidate_item in selected_set:
                continue
            selected_set.add(candidate_item)
            selected_items.append(candidate_item)

        if len(selected_items) == effective_num_samples:
            return selected_items

        available_items = [
            item_id for item_id in self.all_item_ids if item_id not in banned and item_id not in selected_set
        ]
        if not available_items:
            return selected_items
        remaining = min(effective_num_samples - len(selected_items), len(available_items))
        selected_items.extend(sampler.sample(available_items, k=remaining))
        return selected_items


@dataclass
class BatchBuilder:
    user_features: Dict[int, Dict]
    item_features: Dict[int, Dict]
    all_item_ids: Sequence[int]
    user_seen_items: Dict[int, Sequence[int]]
    max_seq_len: int
    max_genres_per_item: int
    seed: int = 42

    def __post_init__(self) -> None:
        self.all_item_ids = list(self.all_item_ids)
        self.user_seen_item_sets = {uid: set(items) for uid, items in self.user_seen_items.items()}
        self.negative_sampler = NegativeSampler(
            all_item_ids=self.all_item_ids,
            user_seen_items=self.user_seen_item_sets,
            seed=self.seed,
        )
        self._warned_eval_caps: set[int] = set()
        self.item_genre_cache = {
            item_id: self._pad_genres(item_feature["genres"]) for item_id, item_feature in self.item_features.items()
        }

    def _warn_eval_negative_cap(self, requested: int, effective: int) -> None:
        if requested in self._warned_eval_caps:
            return
        warnings.warn(
            (
                "Requested more eval negatives than the dataset can provide for some users. "
                f"Auto-truncating from {requested} to {effective} to avoid infinite sampling."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        self._warned_eval_caps.add(requested)

    def _sample_eval_negatives(self, user_id: int, target_item_id: int, num_negatives: int) -> List[int]:
        rng = random.Random(self.seed + user_id * 997 + target_item_id * 101)
        return self.negative_sampler.sample(
            user_id=user_id,
            num_samples=num_negatives,
            banned_items=[target_item_id],
            rng=rng,
        )

    def _pad_genres(self, genres: Sequence[int]) -> Tuple[List[int], List[int]]:
        padded = [0] * self.max_genres_per_item
        mask = [0] * self.max_genres_per_item
        for idx, genre_id in enumerate(genres[: self.max_genres_per_item]):
            padded[idx] = genre_id
            mask[idx] = 1
        return padded, mask

    def _get_item_genres(self, item_id: int) -> Tuple[List[int], List[int]]:
        return self.item_genre_cache[item_id]

    def _build_target_item_tensors(self, candidate_item_ids: Sequence[int]) -> Dict[str, torch.Tensor]:
        target_genres = []
        target_genre_mask = []
        for item_id in candidate_item_ids:
            item_genres, item_genre_mask = self._get_item_genres(item_id)
            target_genres.append(item_genres)
            target_genre_mask.append(item_genre_mask)

        return {
            "target_item_id": torch.tensor(candidate_item_ids, dtype=torch.long),
            "target_genres": torch.tensor(target_genres, dtype=torch.long),
            "target_genre_mask": torch.tensor(target_genre_mask, dtype=torch.float32),
        }

    def _build_shared_sample_tensors(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        user_ids = []
        genders = []
        ages = []
        occupations = []
        target_hours = []
        target_weekdays = []

        hist_item_ids = []
        hist_ratings = []
        hist_hours = []
        hist_weekdays = []
        hist_positions = []
        hist_genres = []
        hist_genre_mask = []
        hist_mask = []

        for sample in samples:
            user_feature = self.user_features[sample["user_id"]]

            user_ids.append(sample["user_id"])
            genders.append(user_feature["gender"])
            ages.append(user_feature["age"])
            occupations.append(user_feature["occupation"])
            target_hours.append(sample["target_hour"])
            target_weekdays.append(sample["target_weekday"])

            seq_items = sample["history_item_ids"][-self.max_seq_len :]
            seq_ratings = sample["history_ratings"][-self.max_seq_len :]
            seq_hours = sample["history_hours"][-self.max_seq_len :]
            seq_weekdays = sample["history_weekdays"][-self.max_seq_len :]
            seq_len = len(seq_items)

            padded_items = [0] * self.max_seq_len
            padded_ratings = [0] * self.max_seq_len
            padded_hours = [0] * self.max_seq_len
            padded_weekdays = [0] * self.max_seq_len
            padded_positions = [0] * self.max_seq_len
            padded_hist_mask = [0] * self.max_seq_len
            padded_hist_genres = [[0] * self.max_genres_per_item for _ in range(self.max_seq_len)]
            padded_hist_genre_mask = [[0] * self.max_genres_per_item for _ in range(self.max_seq_len)]

            for t in range(seq_len):
                item_id = seq_items[t]
                padded_items[t] = item_id
                padded_ratings[t] = seq_ratings[t]
                padded_hours[t] = seq_hours[t]
                padded_weekdays[t] = seq_weekdays[t]
                padded_positions[t] = t + 1
                padded_hist_mask[t] = 1
                movie_genres, movie_genre_mask = self._get_item_genres(item_id)
                padded_hist_genres[t] = movie_genres
                padded_hist_genre_mask[t] = movie_genre_mask

            hist_item_ids.append(padded_items)
            hist_ratings.append(padded_ratings)
            hist_hours.append(padded_hours)
            hist_weekdays.append(padded_weekdays)
            hist_positions.append(padded_positions)
            hist_mask.append(padded_hist_mask)
            hist_genres.append(padded_hist_genres)
            hist_genre_mask.append(padded_hist_genre_mask)

        return {
            "user_id": torch.tensor(user_ids, dtype=torch.long),
            "gender": torch.tensor(genders, dtype=torch.long),
            "age": torch.tensor(ages, dtype=torch.long),
            "occupation": torch.tensor(occupations, dtype=torch.long),
            "target_hour": torch.tensor(target_hours, dtype=torch.long),
            "target_weekday": torch.tensor(target_weekdays, dtype=torch.long),
            "hist_item_ids": torch.tensor(hist_item_ids, dtype=torch.long),
            "hist_ratings": torch.tensor(hist_ratings, dtype=torch.long),
            "hist_hours": torch.tensor(hist_hours, dtype=torch.long),
            "hist_weekdays": torch.tensor(hist_weekdays, dtype=torch.long),
            "hist_positions": torch.tensor(hist_positions, dtype=torch.long),
            "hist_genres": torch.tensor(hist_genres, dtype=torch.long),
            "hist_genre_mask": torch.tensor(hist_genre_mask, dtype=torch.float32),
            "hist_mask": torch.tensor(hist_mask, dtype=torch.bool),
        }

    def _build_feature_tensors(self, samples: Sequence[Dict], candidate_item_ids: Sequence[int] | None = None) -> Dict[str, torch.Tensor]:
        resolved_item_ids = list(candidate_item_ids) if candidate_item_ids is not None else [sample["target_item_id"] for sample in samples]
        batch = self._build_shared_sample_tensors(samples)
        batch.update(self._build_target_item_tensors(resolved_item_ids))
        return batch

    def build_train_batch(self, samples: Sequence[Dict], negative_ratio: int) -> Dict[str, torch.Tensor]:
        pos_batch = self._build_feature_tensors(samples)
        pos_labels = torch.ones(len(samples), dtype=torch.float32)

        if negative_ratio <= 0:
            pos_batch["labels"] = pos_labels
            return pos_batch

        negative_samples: List[Dict] = []
        negative_item_ids: List[int] = []
        for sample in samples:
            negatives = self.negative_sampler.sample(
                user_id=sample["user_id"],
                num_samples=negative_ratio,
                banned_items=[sample["target_item_id"]],
            )
            for neg_item_id in negatives:
                negative_samples.append(sample)
                negative_item_ids.append(neg_item_id)

        neg_batch = self._build_feature_tensors(negative_samples, candidate_item_ids=negative_item_ids)
        neg_labels = torch.zeros(len(negative_samples), dtype=torch.float32)

        merged = {}
        for key in pos_batch:
            merged[key] = torch.cat([pos_batch[key], neg_batch[key]], dim=0)
        merged["labels"] = torch.cat([pos_labels, neg_labels], dim=0)
        return merged

    def build_eval_batch(self, samples: Sequence[Dict], num_negatives: int) -> Tuple[Dict[str, torch.Tensor], int]:
        min_available_negatives = None
        for sample in samples:
            available_count = self.negative_sampler.available_count(
                user_id=sample["user_id"],
                banned_items=[sample["target_item_id"]],
            )
            if min_available_negatives is None:
                min_available_negatives = available_count
            else:
                min_available_negatives = min(min_available_negatives, available_count)

        min_available_negatives = min_available_negatives or 0
        effective_num_negatives = min(num_negatives, min_available_negatives)
        if effective_num_negatives < num_negatives:
            self._warn_eval_negative_cap(requested=num_negatives, effective=effective_num_negatives)
        if effective_num_negatives <= 0:
            raise ValueError("No valid eval negatives are available for at least one sample in the current batch.")

        candidate_item_ids: List[int] = []
        for sample in samples:
            candidate_item_ids.append(sample["target_item_id"])
            for neg_item_id in self._sample_eval_negatives(
                user_id=sample["user_id"],
                target_item_id=sample["target_item_id"],
                num_negatives=effective_num_negatives,
            ):
                candidate_item_ids.append(neg_item_id)

        group_size = effective_num_negatives + 1
        shared_batch = self._build_shared_sample_tensors(samples)
        expanded_batch = {key: value.repeat_interleave(group_size, dim=0) for key, value in shared_batch.items()}
        expanded_batch.update(self._build_target_item_tensors(candidate_item_ids))
        return expanded_batch, group_size
