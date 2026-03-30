#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mixformer.data.preprocess import build_bundle, save_bundle


DEFAULT_RAW_DIRS = {
    "ml-100k": "data/raw/ml-100k",
    "ml-1m": "data/raw/ml-1m",
    "amazon-all-beauty": "data/raw/amazon-all-beauty",
    "amazon-video-games": "data/raw/amazon-video-games",
    "amazon-electronics-x1": "data/raw/amazon-electronics-x1",
    "mind-small": "data/raw/mind-small",
    "taobao-ad-x1": "data/raw/taobao-ad-x1",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess a supported public dataset for MixFormer.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(DEFAULT_RAW_DIRS.keys()),
    )
    parser.add_argument("--raw-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--min-rating", type=int, default=4)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    args = parser.parse_args()

    raw_path = args.raw_path or DEFAULT_RAW_DIRS[args.dataset]
    output_path = args.output_path or f"data/processed/{args.dataset}.pkl"

    bundle = build_bundle(
        dataset_name=args.dataset,
        raw_path=raw_path,
        min_rating=args.min_rating,
        min_user_interactions=args.min_user_interactions,
    )
    save_bundle(bundle, output_path)
    print(f"Saved processed bundle to {output_path}")
    print(bundle["meta"])


if __name__ == "__main__":
    main()
