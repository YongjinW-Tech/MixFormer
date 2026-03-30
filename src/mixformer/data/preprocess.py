from __future__ import annotations

import csv
import gzip
import io
import json
import pickle
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _parse_dat(path: Path, columns: List[str]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=columns,
        encoding="latin-1",
    )


def _parse_hour_and_weekday(timestamp: int) -> tuple[int, int]:
    dt = datetime.utcfromtimestamp(int(timestamp))
    return dt.hour, dt.weekday()


def _parse_movie_year(title: str) -> int:
    match = re.search(r"\((\d{4})\)\s*$", title)
    return int(match.group(1)) if match else 0


def _build_splits_from_grouped(
    grouped: Dict[int, List[Dict]],
    min_user_interactions: int,
) -> tuple[Dict[str, List[Dict]], set[int]]:
    splits = {"train": [], "val": [], "test": []}
    kept_users = set()

    for user_id, interactions in grouped.items():
        if len(interactions) < min_user_interactions:
            continue
        kept_users.add(user_id)

        def build_sample(target_idx: int) -> Dict:
            history = interactions[:target_idx]
            target = interactions[target_idx]
            return {
                "user_id": user_id,
                "target_item_id": target["item_id"],
                "target_rating": target["rating"],
                "target_timestamp": target["timestamp"],
                "target_hour": target["hour"],
                "target_weekday": target["weekday"],
                "history_item_ids": [x["item_id"] for x in history],
                "history_ratings": [x["rating"] for x in history],
                "history_hours": [x["hour"] for x in history],
                "history_weekdays": [x["weekday"] for x in history],
            }

        for target_idx in range(1, len(interactions) - 2):
            splits["train"].append(build_sample(target_idx))
        splits["val"].append(build_sample(len(interactions) - 2))
        splits["test"].append(build_sample(len(interactions) - 1))

    return splits, kept_users


def build_ml1m_bundle(
    raw_dir: str | Path,
    min_rating: int = 4,
    min_user_interactions: int = 5,
) -> Dict:
    raw_dir = Path(raw_dir)
    users = _parse_dat(raw_dir / "users.dat", ["raw_user_id", "gender", "age", "occupation", "zip"])
    movies = _parse_dat(raw_dir / "movies.dat", ["raw_item_id", "title", "genres"])
    ratings = _parse_dat(raw_dir / "ratings.dat", ["raw_user_id", "raw_item_id", "rating", "timestamp"])

    ratings = ratings[ratings["rating"] >= min_rating].copy()

    kept_user_ids = sorted(ratings["raw_user_id"].unique().tolist())
    kept_item_ids = sorted(ratings["raw_item_id"].unique().tolist())

    user_id_map = {raw_id: idx + 1 for idx, raw_id in enumerate(kept_user_ids)}
    item_id_map = {raw_id: idx + 1 for idx, raw_id in enumerate(kept_item_ids)}

    users = users[users["raw_user_id"].isin(user_id_map)].copy()
    movies = movies[movies["raw_item_id"].isin(item_id_map)].copy()
    ratings = ratings[
        ratings["raw_user_id"].isin(user_id_map) & ratings["raw_item_id"].isin(item_id_map)
    ].copy()

    gender_map = {value: idx + 1 for idx, value in enumerate(sorted(users["gender"].unique().tolist()))}
    age_map = {value: idx + 1 for idx, value in enumerate(sorted(users["age"].unique().tolist()))}
    occupation_map = {
        value: idx + 1 for idx, value in enumerate(sorted(users["occupation"].unique().tolist()))
    }

    genre_values = set()
    for genre_string in movies["genres"].tolist():
        for genre in genre_string.split("|"):
            genre_values.add(genre)
    genre_map = {genre: idx + 1 for idx, genre in enumerate(sorted(genre_values))}

    user_features: Dict[int, Dict] = {}
    for row in users.itertuples(index=False):
        user_features[user_id_map[row.raw_user_id]] = {
            "gender": gender_map[row.gender],
            "age": age_map[row.age],
            "occupation": occupation_map[row.occupation],
        }

    item_features: Dict[int, Dict] = {}
    max_genres_per_item = 0
    for row in movies.itertuples(index=False):
        genres = [genre_map[g] for g in row.genres.split("|")]
        max_genres_per_item = max(max_genres_per_item, len(genres))
        item_features[item_id_map[row.raw_item_id]] = {
            "genres": genres,
            "year": _parse_movie_year(row.title),
        }

    ratings["user_id"] = ratings["raw_user_id"].map(user_id_map)
    ratings["item_id"] = ratings["raw_item_id"].map(item_id_map)
    ratings = ratings.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)

    user_seen_items: Dict[int, List[int]] = defaultdict(list)
    grouped = defaultdict(list)
    for row in ratings.itertuples(index=False):
        hour, weekday = _parse_hour_and_weekday(row.timestamp)
        grouped[row.user_id].append(
            {
                "item_id": row.item_id,
                "rating": int(row.rating),
                "timestamp": int(row.timestamp),
                "hour": hour,
                "weekday": weekday,
            }
        )
        user_seen_items[row.user_id].append(row.item_id)

    splits, kept_users = _build_splits_from_grouped(grouped, min_user_interactions=min_user_interactions)

    filtered_user_features = {uid: user_features[uid] for uid in sorted(kept_users)}
    filtered_user_seen_items = {uid: sorted(set(user_seen_items[uid])) for uid in sorted(kept_users)}

    meta = {
        "dataset_name": "ml-1m",
        "num_users": max(filtered_user_features) if filtered_user_features else 0,
        "num_items": max(item_features) if item_features else 0,
        "num_genders": len(gender_map),
        "num_ages": len(age_map),
        "num_occupations": len(occupation_map),
        "num_genres": len(genre_map),
        "num_ratings": int(ratings["rating"].max()),
        "max_genres_per_item": max_genres_per_item,
        "min_rating": min_rating,
        "min_user_interactions": min_user_interactions,
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
    }

    return {
        "meta": meta,
        "user_features": filtered_user_features,
        "item_features": item_features,
        "user_seen_items": filtered_user_seen_items,
        "splits": splits,
    }


def build_ml100k_bundle(
    raw_dir: str | Path,
    min_rating: int = 4,
    min_user_interactions: int = 5,
) -> Dict:
    raw_dir = Path(raw_dir)
    users = pd.read_csv(
        raw_dir / "u.user",
        sep="|",
        header=None,
        names=["raw_user_id", "age", "gender", "occupation", "zip"],
        encoding="latin-1",
    )
    ratings = pd.read_csv(
        raw_dir / "u.data",
        sep="\t",
        header=None,
        names=["raw_user_id", "raw_item_id", "rating", "timestamp"],
        encoding="latin-1",
    )

    genre_columns = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    item_columns = ["raw_item_id", "title", "release_date", "video_release_date", "imdb_url", *genre_columns]
    items = pd.read_csv(
        raw_dir / "u.item",
        sep="|",
        header=None,
        names=item_columns,
        encoding="latin-1",
    )

    ratings = ratings[ratings["rating"] >= min_rating].copy()
    kept_user_ids = sorted(ratings["raw_user_id"].unique().tolist())
    kept_item_ids = sorted(ratings["raw_item_id"].unique().tolist())

    user_id_map = {raw_id: idx + 1 for idx, raw_id in enumerate(kept_user_ids)}
    item_id_map = {raw_id: idx + 1 for idx, raw_id in enumerate(kept_item_ids)}

    users = users[users["raw_user_id"].isin(user_id_map)].copy()
    items = items[items["raw_item_id"].isin(item_id_map)].copy()
    ratings = ratings[
        ratings["raw_user_id"].isin(user_id_map) & ratings["raw_item_id"].isin(item_id_map)
    ].copy()

    gender_map = {value: idx + 1 for idx, value in enumerate(sorted(users["gender"].unique().tolist()))}
    age_map = {value: idx + 1 for idx, value in enumerate(sorted(users["age"].unique().tolist()))}
    occupation_map = {
        value: idx + 1 for idx, value in enumerate(sorted(users["occupation"].unique().tolist()))
    }
    genre_map = {value: idx + 1 for idx, value in enumerate(genre_columns)}

    user_features = {}
    for row in users.itertuples(index=False):
        user_features[user_id_map[row.raw_user_id]] = {
            "gender": gender_map[row.gender],
            "age": age_map[row.age],
            "occupation": occupation_map[row.occupation],
        }

    item_features = {}
    max_genres_per_item = 0
    for _, row in items.iterrows():
        active_genres = []
        for genre_name in genre_columns:
            if int(row[genre_name]) == 1:
                active_genres.append(genre_map[genre_name])
        if not active_genres:
            active_genres = [genre_map["unknown"]]
        max_genres_per_item = max(max_genres_per_item, len(active_genres))
        item_features[item_id_map[int(row["raw_item_id"])]] = {
            "genres": active_genres,
            "year": _parse_movie_year(str(row["title"])),
        }

    ratings["user_id"] = ratings["raw_user_id"].map(user_id_map)
    ratings["item_id"] = ratings["raw_item_id"].map(item_id_map)
    ratings = ratings.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)

    user_seen_items: Dict[int, List[int]] = defaultdict(list)
    grouped = defaultdict(list)
    for row in ratings.itertuples(index=False):
        hour, weekday = _parse_hour_and_weekday(row.timestamp)
        grouped[row.user_id].append(
            {
                "item_id": row.item_id,
                "rating": int(row.rating),
                "timestamp": int(row.timestamp),
                "hour": hour,
                "weekday": weekday,
            }
        )
        user_seen_items[row.user_id].append(row.item_id)

    splits, kept_users = _build_splits_from_grouped(grouped, min_user_interactions=min_user_interactions)
    filtered_user_features = {uid: user_features[uid] for uid in sorted(kept_users)}
    filtered_user_seen_items = {uid: sorted(set(user_seen_items[uid])) for uid in sorted(kept_users)}

    meta = {
        "dataset_name": "ml-100k",
        "num_users": max(filtered_user_features) if filtered_user_features else 0,
        "num_items": max(item_features) if item_features else 0,
        "num_genders": len(gender_map),
        "num_ages": len(age_map),
        "num_occupations": len(occupation_map),
        "num_genres": len(genre_map),
        "num_ratings": int(ratings["rating"].max()),
        "max_genres_per_item": max_genres_per_item,
        "min_rating": min_rating,
        "min_user_interactions": min_user_interactions,
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
    }

    return {
        "meta": meta,
        "user_features": filtered_user_features,
        "item_features": item_features,
        "user_seen_items": filtered_user_seen_items,
        "splits": splits,
    }


def build_amazon_2018_bundle(
    review_path: str | Path,
    meta_path: str | Path,
    dataset_name: str,
    min_rating: int = 4,
    min_user_interactions: int = 5,
) -> Dict:
    review_path = Path(review_path)
    meta_path = Path(meta_path)

    reviews = []
    with gzip.open(review_path, "rt", encoding="utf-8") as fp:
        for line in fp:
            record = json.loads(line)
            rating = int(record.get("overall", 0))
            if rating < min_rating:
                continue
            reviews.append(
                {
                    "raw_user_id": record["reviewerID"],
                    "raw_item_id": record["asin"],
                    "rating": rating,
                    "timestamp": int(record["unixReviewTime"]),
                }
            )

    ratings = pd.DataFrame(reviews)
    kept_user_ids = sorted(ratings["raw_user_id"].unique().tolist())
    kept_item_ids = sorted(ratings["raw_item_id"].unique().tolist())
    user_id_map = {raw_id: idx + 1 for idx, raw_id in enumerate(kept_user_ids)}
    item_id_map = {raw_id: idx + 1 for idx, raw_id in enumerate(kept_item_ids)}

    category_tokens = {"unknown"}
    raw_item_categories: Dict[str, List[str]] = {}
    with gzip.open(meta_path, "rt", encoding="utf-8") as fp:
        for line in fp:
            record = json.loads(line)
            asin = record.get("asin")
            if asin not in item_id_map:
                continue
            categories = record.get("categories")
            if categories:
                flat_categories = sorted({token for path in categories for token in path}) or ["unknown"]
            else:
                category = record.get("category") or []
                flat_categories = sorted(set(category)) or ["unknown"]
            raw_item_categories[asin] = flat_categories
            category_tokens.update(flat_categories)

    genre_map = {value: idx + 1 for idx, value in enumerate(sorted(category_tokens))}
    item_features = {}
    max_genres_per_item = 0
    for raw_item_id, item_id in item_id_map.items():
        categories = raw_item_categories.get(raw_item_id, ["unknown"])
        genre_ids = [genre_map[token] for token in categories]
        max_genres_per_item = max(max_genres_per_item, len(genre_ids))
        item_features[item_id] = {"genres": genre_ids, "year": 0}

    user_features = {
        user_id_map[raw_user_id]: {"gender": 1, "age": 1, "occupation": 1} for raw_user_id in kept_user_ids
    }

    ratings["user_id"] = ratings["raw_user_id"].map(user_id_map)
    ratings["item_id"] = ratings["raw_item_id"].map(item_id_map)
    ratings = ratings.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)

    user_seen_items: Dict[int, List[int]] = defaultdict(list)
    grouped = defaultdict(list)
    for row in ratings.itertuples(index=False):
        hour, weekday = _parse_hour_and_weekday(row.timestamp)
        grouped[row.user_id].append(
            {
                "item_id": row.item_id,
                "rating": int(row.rating),
                "timestamp": int(row.timestamp),
                "hour": hour,
                "weekday": weekday,
            }
        )
        user_seen_items[row.user_id].append(row.item_id)

    splits, kept_users = _build_splits_from_grouped(grouped, min_user_interactions=min_user_interactions)
    filtered_user_features = {uid: user_features[uid] for uid in sorted(kept_users)}
    filtered_user_seen_items = {uid: sorted(set(user_seen_items[uid])) for uid in sorted(kept_users)}

    meta = {
        "dataset_name": dataset_name,
        "num_users": max(filtered_user_features) if filtered_user_features else 0,
        "num_items": max(item_features) if item_features else 0,
        "num_genders": 1,
        "num_ages": 1,
        "num_occupations": 1,
        "num_genres": len(genre_map),
        "num_ratings": int(ratings["rating"].max()),
        "max_genres_per_item": max_genres_per_item,
        "min_rating": min_rating,
        "min_user_interactions": min_user_interactions,
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
    }

    return {
        "meta": meta,
        "user_features": filtered_user_features,
        "item_features": item_features,
        "user_seen_items": filtered_user_seen_items,
        "splits": splits,
    }


def _split_history_field(value: str, delimiter: str = "^") -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [token for token in text.split(delimiter) if token]


def _normalize_sparse_token(value: str, prefix: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        cleaned = "unknown"
    return f"{prefix}:{cleaned}"


def _bucket_taobao_price(value: str) -> str:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return "price_bucket:unknown"
    bucket = max(0, min(int(price * 20), 19))
    return f"price_bucket:{bucket}"


def _iter_taobao_rows(raw_path: str | Path, split_name: str):
    raw_path = Path(raw_path)
    if raw_path.is_dir():
        zip_path = raw_path / "TaobaoAd_x1.zip"
        if zip_path.exists():
            raw_path = zip_path

    if raw_path.is_file():
        with zipfile.ZipFile(raw_path) as zip_fp:
            with zip_fp.open(f"{split_name}.csv") as member_fp:
                text_fp = io.TextIOWrapper(member_fp, encoding="utf-8")
                next(text_fp, None)
                for line in text_fp:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    yield line.split(",")
        return

    csv_path = raw_path / f"{split_name}.csv"
    with open(csv_path, "r", encoding="utf-8") as fp:
        next(fp, None)
        for line in fp:
            line = line.rstrip("\n")
            if not line:
                continue
            yield line.split(",")


def _collect_reczoo_amazon_samples(
    csv_path: Path,
) -> tuple[List[Dict], Dict[str, List[str]], set[str], Dict[str, str]]:
    samples: List[Dict] = []
    user_seen_items: Dict[str, List[str]] = defaultdict(list)
    raw_user_ids: set[str] = set()
    raw_item_to_cate: Dict[str, str] = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            raw_user_id = str(row["user_id"])
            raw_item_id = str(row["item_id"])
            raw_cate_id = str(row.get("cate_id", "")).strip() or "unknown"

            raw_user_ids.add(raw_user_id)
            raw_item_to_cate[raw_item_id] = raw_cate_id

            history_items = _split_history_field(row.get("item_history", ""))
            history_cates = _split_history_field(row.get("cate_history", ""))
            for history_item_id, history_cate_id in zip(history_items, history_cates):
                if history_item_id:
                    raw_item_to_cate.setdefault(history_item_id, history_cate_id or "unknown")

            if history_items:
                user_seen_items[raw_user_id].extend(history_items)

            if int(row["label"]) != 1:
                continue

            samples.append(
                {
                    "raw_user_id": raw_user_id,
                    "raw_target_item_id": raw_item_id,
                    "target_rating": 1,
                    "target_hour": 0,
                    "target_weekday": 0,
                    "history_item_ids": history_items,
                    "history_ratings": [1] * len(history_items),
                    "history_hours": [0] * len(history_items),
                    "history_weekdays": [0] * len(history_items),
                }
            )
            user_seen_items[raw_user_id].append(raw_item_id)

    return samples, user_seen_items, raw_user_ids, raw_item_to_cate


def build_amazon_electronics_x1_bundle(
    raw_dir: str | Path,
    min_rating: int = 1,
    min_user_interactions: int = 5,
) -> Dict:
    raw_dir = Path(raw_dir)
    train_samples, train_seen_items, train_users, train_item_to_cate = _collect_reczoo_amazon_samples(
        raw_dir / "train.csv"
    )
    test_samples, test_seen_items, test_users, test_item_to_cate = _collect_reczoo_amazon_samples(
        raw_dir / "test.csv"
    )

    raw_user_ids = sorted(train_users | test_users, key=int)
    raw_item_to_cate = {**train_item_to_cate, **test_item_to_cate}
    raw_item_ids = sorted(raw_item_to_cate, key=int)

    user_id_map = {raw_id: idx + 1 for idx, raw_id in enumerate(raw_user_ids)}
    item_id_map = {raw_id: idx + 1 for idx, raw_id in enumerate(raw_item_ids)}

    raw_cate_values = sorted({cate_id or "unknown" for cate_id in raw_item_to_cate.values()}, key=str)
    genre_map = {raw_cate_id: idx + 1 for idx, raw_cate_id in enumerate(raw_cate_values)}

    item_features = {
        item_id_map[raw_item_id]: {"genres": [genre_map[raw_item_to_cate[raw_item_id]]], "year": 0}
        for raw_item_id in raw_item_ids
    }

    all_user_seen_items: Dict[int, List[int]] = defaultdict(list)
    for source in [train_seen_items, test_seen_items]:
        for raw_user_id, raw_items in source.items():
            mapped_user_id = user_id_map[raw_user_id]
            all_user_seen_items[mapped_user_id].extend(
                item_id_map[item_id]
                for item_id in raw_items
                if item_id in item_id_map
            )

    user_features = {
        user_id_map[raw_user_id]: {"gender": 1, "age": 1, "occupation": 1}
        for raw_user_id in raw_user_ids
    }

    user_interaction_counts = {user_id: len(set(items)) for user_id, items in all_user_seen_items.items()}
    kept_user_ids = {
        user_id
        for user_id, count in user_interaction_counts.items()
        if count >= min_user_interactions
    }

    def convert_sample(sample: Dict) -> Dict:
        history_item_ids = [
            item_id_map[item_id]
            for item_id in sample["history_item_ids"]
            if item_id in item_id_map
        ]
        seq_len = len(history_item_ids)
        return {
            "user_id": user_id_map[sample["raw_user_id"]],
            "target_item_id": item_id_map[sample["raw_target_item_id"]],
            "target_rating": sample["target_rating"],
            "target_timestamp": 0,
            "target_hour": sample["target_hour"],
            "target_weekday": sample["target_weekday"],
            "history_item_ids": history_item_ids,
            "history_ratings": sample["history_ratings"][:seq_len],
            "history_hours": sample["history_hours"][:seq_len],
            "history_weekdays": sample["history_weekdays"][:seq_len],
        }

    grouped_train_samples: Dict[int, List[Dict]] = defaultdict(list)
    for sample in train_samples:
        mapped_user_id = user_id_map[sample["raw_user_id"]]
        if mapped_user_id not in kept_user_ids:
            continue
        converted = convert_sample(sample)
        if not converted["history_item_ids"]:
            continue
        grouped_train_samples[converted["user_id"]].append(converted)

    final_train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    for user_id in sorted(grouped_train_samples):
        samples = grouped_train_samples[user_id]
        if len(samples) == 1:
            final_train_samples.extend(samples)
            continue
        final_train_samples.extend(samples[:-1])
        val_samples.append(samples[-1])

    final_test_samples: List[Dict] = []
    for sample in test_samples:
        mapped_user_id = user_id_map[sample["raw_user_id"]]
        if mapped_user_id not in kept_user_ids:
            continue
        converted = convert_sample(sample)
        if not converted["history_item_ids"]:
            continue
        final_test_samples.append(converted)

    if not val_samples and final_train_samples:
        val_samples.append(final_train_samples.pop())
    if not final_test_samples and val_samples:
        final_test_samples.append(val_samples[-1])

    filtered_user_features = {uid: user_features[uid] for uid in sorted(kept_user_ids)}
    filtered_user_seen_items = {
        uid: sorted(set(all_user_seen_items[uid]))
        for uid in sorted(kept_user_ids)
    }

    meta = {
        "dataset_name": "amazon-electronics-x1",
        "num_users": max(filtered_user_features) if filtered_user_features else 0,
        "num_items": max(item_features) if item_features else 0,
        "num_genders": 1,
        "num_ages": 1,
        "num_occupations": 1,
        "num_genres": len(genre_map),
        "num_ratings": 1,
        "max_genres_per_item": 1,
        "min_rating": 1,
        "min_user_interactions": min_user_interactions,
        "train_size": len(final_train_samples),
        "val_size": len(val_samples),
        "test_size": len(final_test_samples),
    }

    return {
        "meta": meta,
        "user_features": filtered_user_features,
        "item_features": item_features,
        "user_seen_items": filtered_user_seen_items,
        "candidate_item_ids": sorted(item_id_map[item_id] for item_id in raw_item_ids),
        "splits": {
            "train": final_train_samples,
            "val": val_samples,
            "test": final_test_samples,
        },
    }


def build_taobao_ad_x1_bundle(
    raw_path: str | Path,
    min_rating: int = 1,
    min_user_interactions: int = 5,
) -> Dict:
    del min_rating
    raw_path = Path(raw_path)

    gender_map: Dict[str, int] = {}
    age_map: Dict[str, int] = {}
    occupation_map: Dict[str, int] = {}
    history_rating_map: Dict[str, int] = {}
    genre_token_map: Dict[str, int] = {}
    user_id_map: Dict[str, int] = {}
    item_id_map: Dict[str, int] = {}

    user_features: Dict[int, Dict] = {}
    item_features: Dict[int, Dict] = {}
    user_positive_targets: Dict[int, set[int]] = defaultdict(set)
    candidate_item_ids: set[int] = set()
    max_genres_per_item = 0

    def map_value(mapping: Dict[str, int], raw_value: str) -> int:
        key = str(raw_value or "").strip() or "unknown"
        mapped = mapping.get(key)
        if mapped is None:
            mapped = len(mapping) + 1
            mapping[key] = mapped
        return mapped

    def map_user(
        raw_user_id: str,
        gender_value: str,
        age_value: str,
        occupation_value: str,
    ) -> int:
        mapped_user_id = user_id_map.get(raw_user_id)
        if mapped_user_id is None:
            mapped_user_id = len(user_id_map) + 1
            user_id_map[raw_user_id] = mapped_user_id

        user_features[mapped_user_id] = {
            "gender": map_value(gender_map, gender_value),
            "age": map_value(age_map, age_value),
            "occupation": map_value(occupation_map, occupation_value),
        }
        return mapped_user_id

    def register_item(raw_item_key: str, feature_tokens: List[str]) -> int:
        nonlocal max_genres_per_item
        mapped_item_id = item_id_map.get(raw_item_key)
        if mapped_item_id is None:
            mapped_item_id = len(item_id_map) + 1
            item_id_map[raw_item_key] = mapped_item_id
            deduped_tokens = list(dict.fromkeys(feature_tokens)) or ["feature:unknown"]
            genre_ids = [map_value(genre_token_map, token) for token in deduped_tokens]
            item_features[mapped_item_id] = {"genres": genre_ids, "year": 0}
            max_genres_per_item = max(max_genres_per_item, len(genre_ids))
        return mapped_item_id

    for split_name in ["train", "test"]:
        for row in _iter_taobao_rows(raw_path, split_name):
            if len(row) < 21:
                continue
            clk = row[0]
            btag_his = row[1]
            cate_his = row[2]
            userid = row[4]
            final_gender_code = row[7]
            age_level = row[8]
            occupation = row[11]
            adgroup_id = row[13]
            cate_id = row[14]
            campaign_id = row[15]
            customer = row[16]
            brand = row[17]
            price = row[18]
            pid = row[19]
            btag = row[20]

            mapped_user_id = map_user(
                raw_user_id=userid,
                gender_value=final_gender_code,
                age_value=age_level,
                occupation_value=occupation,
            )

            target_item_id = register_item(
                raw_item_key=f"ad:{adgroup_id}",
                feature_tokens=[
                    _normalize_sparse_token(cate_id, "cate"),
                    _normalize_sparse_token(brand, "brand"),
                    _normalize_sparse_token(campaign_id, "campaign"),
                    _normalize_sparse_token(customer, "customer"),
                    _normalize_sparse_token(pid, "pid"),
                    _normalize_sparse_token(btag, "btag"),
                    _bucket_taobao_price(price),
                ],
            )
            candidate_item_ids.add(target_item_id)

            if clk != "1":
                continue

            user_positive_targets[mapped_user_id].add(target_item_id)

            for raw_history_btag in _split_history_field(btag_his):
                map_value(history_rating_map, raw_history_btag)

            for raw_history_cate in _split_history_field(cate_his):
                register_item(
                    raw_item_key=f"hist_cate:{raw_history_cate}",
                    feature_tokens=[_normalize_sparse_token(raw_history_cate, "cate")],
                )

    kept_user_ids = {
        user_id
        for user_id, seen_targets in user_positive_targets.items()
        if len(seen_targets) >= min_user_interactions
    }

    def build_sample(row: List[str]) -> Dict | None:
        if len(row) < 21 or row[0] != "1":
            return None

        userid = row[4]
        mapped_user_id = user_id_map.get(userid)
        if mapped_user_id is None or mapped_user_id not in kept_user_ids:
            return None

        history_cates = _split_history_field(row[2])
        history_btags = _split_history_field(row[1])
        history_item_ids = []
        history_ratings = []
        for idx, raw_history_cate in enumerate(history_cates):
            history_item_id = item_id_map.get(f"hist_cate:{raw_history_cate}")
            if history_item_id is None:
                continue
            history_item_ids.append(history_item_id)
            raw_history_btag = history_btags[idx] if idx < len(history_btags) else ""
            history_ratings.append(history_rating_map.get(raw_history_btag, 0))

        if not history_item_ids:
            return None

        return {
            "user_id": mapped_user_id,
            "target_item_id": item_id_map[f"ad:{row[13]}"],
            "target_rating": 1,
            "target_timestamp": 0,
            "target_hour": 0,
            "target_weekday": 0,
            "history_item_ids": history_item_ids,
            "history_ratings": history_ratings,
            "history_hours": [0] * len(history_item_ids),
            "history_weekdays": [0] * len(history_item_ids),
        }

    final_train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    pending_last_train: Dict[int, Dict] = {}
    for row in _iter_taobao_rows(raw_path, "train"):
        sample = build_sample(row)
        if sample is None:
            continue
        previous_sample = pending_last_train.get(sample["user_id"])
        if previous_sample is not None:
            final_train_samples.append(previous_sample)
        pending_last_train[sample["user_id"]] = sample

    val_samples.extend(pending_last_train[user_id] for user_id in sorted(pending_last_train))
    if not val_samples and final_train_samples:
        val_samples.append(final_train_samples.pop())

    final_test_samples: List[Dict] = []
    for row in _iter_taobao_rows(raw_path, "test"):
        sample = build_sample(row)
        if sample is not None:
            final_test_samples.append(sample)
    if not final_test_samples and val_samples:
        final_test_samples.append(val_samples[-1])

    filtered_user_features = {uid: user_features[uid] for uid in sorted(kept_user_ids)}
    filtered_user_seen_items = {
        uid: sorted(user_positive_targets[uid])
        for uid in sorted(kept_user_ids)
    }

    meta = {
        "dataset_name": "taobao-ad-x1",
        "num_users": max(filtered_user_features) if filtered_user_features else 0,
        "num_items": max(item_features) if item_features else 0,
        "num_genders": len(gender_map),
        "num_ages": len(age_map),
        "num_occupations": len(occupation_map),
        "num_genres": len(genre_token_map),
        "num_ratings": max(len(history_rating_map), 1),
        "max_genres_per_item": max_genres_per_item,
        "min_rating": 1,
        "min_user_interactions": min_user_interactions,
        "train_size": len(final_train_samples),
        "val_size": len(val_samples),
        "test_size": len(final_test_samples),
    }

    return {
        "meta": meta,
        "user_features": filtered_user_features,
        "item_features": item_features,
        "user_seen_items": filtered_user_seen_items,
        "candidate_item_ids": sorted(candidate_item_ids),
        "splits": {
            "train": final_train_samples,
            "val": val_samples,
            "test": final_test_samples,
        },
    }


def _parse_mind_time(timestamp: str) -> tuple[int, int]:
    dt = datetime.strptime(timestamp.strip(), "%m/%d/%Y %I:%M:%S %p")
    return dt.hour, dt.weekday()


def _parse_mind_hour_token(value: str) -> int:
    match = re.fullmatch(r"(\d{1,2})(AM|PM)", value.strip().upper())
    if match is None:
        return 0
    hour = int(match.group(1)) % 12
    if match.group(2) == "PM":
        hour += 12
    return hour


def _normalize_mind_category(value: str, prefix: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        cleaned = "unknown"
    return f"{prefix}:{cleaned.lower()}"


def _load_mind_news_frames(raw_dir: Path) -> List[pd.DataFrame]:
    news_columns = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    frames = []
    for split_name in ["train", "dev"]:
        news_path = raw_dir / split_name / "news.tsv"
        frames.append(
            pd.read_csv(
                news_path,
                sep="\t",
                header=None,
                names=news_columns,
                usecols=["news_id", "category", "subcategory"],
                encoding="utf-8",
                keep_default_na=False,
            )
        )
    return frames


def _collect_mind_samples(
    behaviors_path: Path,
    split_name: str,
) -> tuple[List[Dict], Dict[str, List[str]], set[str], set[str]]:
    samples: List[Dict] = []
    user_seen_items: Dict[str, List[str]] = defaultdict(list)
    raw_user_ids: set[str] = set()
    raw_item_ids: set[str] = set()

    with open(behaviors_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 5:
                continue

            _, raw_user_id, timestamp, history_text, impressions_text = fields
            raw_user_ids.add(raw_user_id)
            hour, weekday = _parse_mind_time(timestamp)

            history_items = [token for token in history_text.split() if token]
            raw_item_ids.update(history_items)
            if history_items:
                user_seen_items[raw_user_id].extend(history_items)

            clicked_items = []
            for impression in impressions_text.split():
                if "-" not in impression:
                    continue
                raw_item_id, label = impression.rsplit("-", 1)
                if not raw_item_id:
                    continue
                raw_item_ids.add(raw_item_id)
                if label == "1":
                    clicked_items.append(raw_item_id)

            for raw_item_id in clicked_items:
                samples.append(
                    {
                        "split_name": split_name,
                        "raw_user_id": raw_user_id,
                        "raw_target_item_id": raw_item_id,
                        "target_rating": 1,
                        "target_hour": hour,
                        "target_weekday": weekday,
                        "history_item_ids": history_items,
                        "history_ratings": [1] * len(history_items),
                        "history_hours": [hour] * len(history_items),
                        "history_weekdays": [weekday] * len(history_items),
                    }
                )
                user_seen_items[raw_user_id].append(raw_item_id)

    return samples, user_seen_items, raw_user_ids, raw_item_ids


def _load_mind_small_news_corpus(
    news_path: Path,
) -> tuple[Dict[str, Dict], set[str], set[str]]:
    raw_item_features: Dict[str, Dict] = {}
    raw_item_ids: set[str] = set()
    category_tokens = {
        _normalize_mind_category("unknown", "cat"),
        _normalize_mind_category("unknown", "subcat"),
    }

    with open(news_path, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            raw_item_id = row["news_id"]
            if not raw_item_id:
                continue
            raw_item_ids.add(raw_item_id)
            category_token = _normalize_mind_category(row.get("cat", ""), "cat")
            subcategory_token = _normalize_mind_category(row.get("sub_cat", ""), "subcat")
            category_tokens.update([category_token, subcategory_token])
            raw_item_features[raw_item_id] = {
                "tokens": [category_token, subcategory_token],
                "year": 0,
            }

    return raw_item_features, raw_item_ids, category_tokens


def _collect_mind_small_samples_from_csv(
    csv_path: Path,
    split_name: str,
) -> tuple[List[Dict], Dict[str, List[str]], set[str], set[str]]:
    samples: List[Dict] = []
    user_seen_items: Dict[str, List[str]] = defaultdict(list)
    raw_user_ids: set[str] = set()
    raw_item_ids: set[str] = set()

    with open(csv_path, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            raw_user_id = row["user_id"]
            raw_item_id = row["news_id"]
            raw_user_ids.add(raw_user_id)
            raw_item_ids.add(raw_item_id)

            history_items = [token for token in str(row.get("news_his", "")).split("^") if token]
            raw_item_ids.update(history_items)
            if history_items:
                user_seen_items[raw_user_id].extend(history_items)

            if int(row["click"]) != 1:
                continue

            hour = _parse_mind_hour_token(str(row.get("hour", "0AM")))
            samples.append(
                {
                    "split_name": split_name,
                    "raw_user_id": raw_user_id,
                    "raw_target_item_id": raw_item_id,
                    "target_rating": 1,
                    "target_hour": hour,
                    "target_weekday": 0,
                    "history_item_ids": history_items,
                    "history_ratings": [1] * len(history_items),
                    "history_hours": [hour] * len(history_items),
                    "history_weekdays": [0] * len(history_items),
                }
            )
            user_seen_items[raw_user_id].append(raw_item_id)

    return samples, user_seen_items, raw_user_ids, raw_item_ids


def build_mind_small_bundle(
    raw_dir: str | Path,
    min_rating: int = 1,
    min_user_interactions: int = 5,
) -> Dict:
    raw_dir = Path(raw_dir)
    mirror_news_path = raw_dir / "news_corpus.tsv"
    if mirror_news_path.exists():
        raw_item_features, raw_item_tokens, category_tokens = _load_mind_small_news_corpus(mirror_news_path)
        train_samples, train_seen_items, train_users, train_items = _collect_mind_small_samples_from_csv(
            raw_dir / "train.csv",
            split_name="train",
        )
        dev_samples, dev_seen_items, dev_users, dev_items = _collect_mind_small_samples_from_csv(
            raw_dir / "valid.csv",
            split_name="dev",
        )
    else:
        news_frames = _load_mind_news_frames(raw_dir)

        raw_item_tokens: set[str] = set()
        category_tokens = {
            _normalize_mind_category("unknown", "cat"),
            _normalize_mind_category("unknown", "subcat"),
        }
        raw_item_features: Dict[str, Dict] = {}
        for frame in news_frames:
            for row in frame.itertuples(index=False):
                raw_item_id = row.news_id
                raw_item_tokens.add(raw_item_id)
                category_token = _normalize_mind_category(row.category, "cat")
                subcategory_token = _normalize_mind_category(row.subcategory, "subcat")
                category_tokens.update([category_token, subcategory_token])
                raw_item_features[raw_item_id] = {
                    "tokens": [category_token, subcategory_token],
                    "year": 0,
                }

        train_samples, train_seen_items, train_users, train_items = _collect_mind_samples(
            raw_dir / "train" / "behaviors.tsv",
            split_name="train",
        )
        dev_samples, dev_seen_items, dev_users, dev_items = _collect_mind_samples(
            raw_dir / "dev" / "behaviors.tsv",
            split_name="dev",
        )

    raw_user_ids = sorted(train_users | dev_users)
    raw_item_ids = sorted(raw_item_tokens | train_items | dev_items)
    user_id_map = {raw_id: idx + 1 for idx, raw_id in enumerate(raw_user_ids)}
    item_id_map = {raw_id: idx + 1 for idx, raw_id in enumerate(raw_item_ids)}

    genre_map = {token: idx + 1 for idx, token in enumerate(sorted(category_tokens))}
    default_tokens = [
        _normalize_mind_category("unknown", "cat"),
        _normalize_mind_category("unknown", "subcat"),
    ]

    item_features: Dict[int, Dict] = {}
    for raw_item_id in raw_item_ids:
        tokens = raw_item_features.get(raw_item_id, {"tokens": default_tokens, "year": 0})["tokens"]
        # MIND 里最稳妥的结构化 item 特征是 category / subcategory，
        # 这里映射到当前复现代码里的 multi-hot genres 槽位。
        ordered_tokens = list(dict.fromkeys(tokens))
        item_features[item_id_map[raw_item_id]] = {
            "genres": [genre_map[token] for token in ordered_tokens],
            "year": 0,
        }

    def convert_sample(sample: Dict) -> Dict:
        history_item_ids = [
            item_id_map[item_id]
            for item_id in sample["history_item_ids"]
            if item_id in item_id_map
        ]
        seq_len = len(history_item_ids)
        return {
            "user_id": user_id_map[sample["raw_user_id"]],
            "target_item_id": item_id_map[sample["raw_target_item_id"]],
            "target_rating": sample["target_rating"],
            "target_timestamp": 0,
            "target_hour": sample["target_hour"],
            "target_weekday": sample["target_weekday"],
            "history_item_ids": history_item_ids,
            "history_ratings": sample["history_ratings"][:seq_len],
            "history_hours": sample["history_hours"][:seq_len],
            "history_weekdays": sample["history_weekdays"][:seq_len],
        }

    all_user_seen_items: Dict[int, List[int]] = defaultdict(list)
    for source in [train_seen_items, dev_seen_items]:
        for raw_user_id, raw_items in source.items():
            mapped_user_id = user_id_map[raw_user_id]
            all_user_seen_items[mapped_user_id].extend(
                item_id_map[item_id] for item_id in raw_items if item_id in item_id_map
            )

    user_features = {
        user_id_map[raw_user_id]: {"gender": 1, "age": 1, "occupation": 1}
        for raw_user_id in raw_user_ids
    }

    user_interaction_counts = {
        user_id: len(set(items))
        for user_id, items in all_user_seen_items.items()
    }
    kept_user_ids = {
        user_id
        for user_id, count in user_interaction_counts.items()
        if count >= min_user_interactions
    }

    converted_train = []
    for sample in train_samples:
        if user_id_map[sample["raw_user_id"]] not in kept_user_ids:
            continue
        if sample["raw_target_item_id"] not in item_id_map:
            continue
        converted = convert_sample(sample)
        if not converted["history_item_ids"]:
            continue
        converted_train.append(converted)

    converted_dev = []
    for sample in dev_samples:
        if user_id_map[sample["raw_user_id"]] not in kept_user_ids:
            continue
        if sample["raw_target_item_id"] not in item_id_map:
            continue
        converted = convert_sample(sample)
        if not converted["history_item_ids"]:
            continue
        converted_dev.append(converted)

    dev_by_user: Dict[int, List[Dict]] = defaultdict(list)
    for sample in converted_dev:
        dev_by_user[sample["user_id"]].append(sample)

    val_samples: List[Dict] = []
    test_samples: List[Dict] = []
    singleton_samples: List[Dict] = []
    for user_id in sorted(dev_by_user):
        samples = dev_by_user[user_id]
        if len(samples) == 1:
            singleton_samples.extend(samples)
            continue
        val_samples.extend(samples[:-1])
        test_samples.append(samples[-1])

    for idx, sample in enumerate(singleton_samples):
        if idx % 2 == 0:
            val_samples.append(sample)
        else:
            test_samples.append(sample)

    if not val_samples and test_samples:
        val_samples.append(test_samples.pop())
    if not test_samples and val_samples:
        test_samples.append(val_samples.pop())

    filtered_user_features = {uid: user_features[uid] for uid in sorted(kept_user_ids)}
    filtered_user_seen_items = {
        uid: sorted(set(all_user_seen_items[uid]))
        for uid in sorted(kept_user_ids)
    }

    max_genres_per_item = max((len(item_feature["genres"]) for item_feature in item_features.values()), default=1)
    meta = {
        "dataset_name": "mind-small",
        "num_users": max(filtered_user_features) if filtered_user_features else 0,
        "num_items": max(item_features) if item_features else 0,
        "num_genders": 1,
        "num_ages": 1,
        "num_occupations": 1,
        "num_genres": len(genre_map),
        "num_ratings": 1,
        "max_genres_per_item": max_genres_per_item,
        "min_rating": 1,
        "min_user_interactions": min_user_interactions,
        "train_size": len(converted_train),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
    }

    return {
        "meta": meta,
        "user_features": filtered_user_features,
        "item_features": item_features,
        "user_seen_items": filtered_user_seen_items,
        "splits": {
            "train": converted_train,
            "val": val_samples,
            "test": test_samples,
        },
    }


def build_bundle(
    dataset_name: str,
    raw_path: str | Path,
    min_rating: int = 4,
    min_user_interactions: int = 5,
) -> Dict:
    raw_path = Path(raw_path)
    if dataset_name == "ml-1m":
        return build_ml1m_bundle(
            raw_dir=raw_path,
            min_rating=min_rating,
            min_user_interactions=min_user_interactions,
        )
    if dataset_name == "ml-100k":
        return build_ml100k_bundle(
            raw_dir=raw_path,
            min_rating=min_rating,
            min_user_interactions=min_user_interactions,
        )
    if dataset_name == "amazon-all-beauty":
        return build_amazon_2018_bundle(
            review_path=raw_path / "All_Beauty_5.json.gz",
            meta_path=raw_path / "meta_All_Beauty.json.gz",
            dataset_name=dataset_name,
            min_rating=min_rating,
            min_user_interactions=min_user_interactions,
        )
    if dataset_name == "amazon-video-games":
        return build_amazon_2018_bundle(
            review_path=raw_path / "Video_Games_5.json.gz",
            meta_path=raw_path / "meta_Video_Games.json.gz",
            dataset_name=dataset_name,
            min_rating=min_rating,
            min_user_interactions=min_user_interactions,
        )
    if dataset_name == "amazon-electronics-x1":
        return build_amazon_electronics_x1_bundle(
            raw_dir=raw_path,
            min_rating=min_rating,
            min_user_interactions=min_user_interactions,
        )
    if dataset_name == "mind-small":
        return build_mind_small_bundle(
            raw_dir=raw_path,
            min_rating=min_rating,
            min_user_interactions=min_user_interactions,
        )
    if dataset_name == "taobao-ad-x1":
        return build_taobao_ad_x1_bundle(
            raw_path=raw_path,
            min_rating=min_rating,
            min_user_interactions=min_user_interactions,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def save_bundle(bundle: Dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fp:
        pickle.dump(bundle, fp)


def load_bundle(path: str | Path) -> Dict:
    with open(path, "rb") as fp:
        return pickle.load(fp)
