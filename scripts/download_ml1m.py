#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path


DATASET_URLS = [
    "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MovieLens-1M.")
    parser.add_argument("--output-dir", type=str, default="data/raw")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "ml-1m.zip"
    extract_dir = output_dir / "ml-1m"

    if extract_dir.exists():
        print(f"Dataset already extracted at {extract_dir}")
        return

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    last_error = None
    for dataset_url in DATASET_URLS:
        try:
            print(f"Downloading {dataset_url} -> {zip_path}")
            request = urllib.request.Request(dataset_url, headers={"User-Agent": "MixFormer-Reproduction/1.0"})
            with urllib.request.urlopen(request, context=ssl_context) as response, open(zip_path, "wb") as fp:
                shutil.copyfileobj(response, fp)
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"Download attempt failed for {dataset_url}: {exc}")
    if last_error is not None:
        raise last_error

    print(f"Extracting {zip_path} -> {output_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_fp:
        zip_fp.extractall(output_dir)

    print(f"Done. Raw files are available under {extract_dir}")


if __name__ == "__main__":
    main()
