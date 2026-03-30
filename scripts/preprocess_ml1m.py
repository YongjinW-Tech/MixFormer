#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mixformer.data.preprocess import build_ml1m_bundle, save_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MovieLens-1M for MixFormer.")
    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="data/processed/ml1m_mixformer.pkl")
    parser.add_argument("--min-rating", type=int, default=4)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    args = parser.parse_args()

    bundle = build_ml1m_bundle(
        raw_dir=args.raw_dir,
        min_rating=args.min_rating,
        min_user_interactions=args.min_user_interactions,
    )
    save_bundle(bundle, args.output_path)
    print(f"Saved processed bundle to {args.output_path}")
    print(bundle["meta"])


if __name__ == "__main__":
    main()
