#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path


DATASET_SPECS = {
    "ml-100k": {
        "kind": "zip",
        "target_dir": "ml-100k",
        "files": [("ml-100k.zip", "https://files.grouplens.org/datasets/movielens/ml-100k.zip")],
    },
    "ml-1m": {
        "kind": "zip",
        "target_dir": "ml-1m",
        "files": [("ml-1m.zip", "https://files.grouplens.org/datasets/movielens/ml-1m.zip")],
    },
    "amazon-all-beauty": {
        "kind": "file",
        "target_dir": "amazon-all-beauty",
        "files": [
            (
                "All_Beauty_5.json.gz",
                "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/All_Beauty_5.json.gz",
            ),
            (
                "meta_All_Beauty.json.gz",
                "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_All_Beauty.json.gz",
            ),
        ],
    },
    "amazon-video-games": {
        "kind": "file",
        "target_dir": "amazon-video-games",
        "files": [
            (
                "Video_Games_5.json.gz",
                "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz",
            ),
            (
                "meta_Video_Games.json.gz",
                "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz",
            ),
        ],
    },
    "amazon-electronics-x1": {
        "kind": "zip_flat",
        "target_dir": "amazon-electronics-x1",
        "files": [
            (
                "AmazonElectronics_x1.zip",
                "https://huggingface.co/datasets/reczoo/AmazonElectronics_x1/resolve/main/AmazonElectronics_x1.zip",
            )
        ],
    },
    "mind-small": {
        "kind": "zip_flat",
        "target_dir": "mind-small",
        "files": [
            (
                "MIND_small_x1.zip",
                "https://huggingface.co/datasets/reczoo/MIND_small_x1/resolve/main/MIND_small_x1.zip",
            )
        ],
    },
    "taobao-ad-x1": {
        "kind": "file",
        "target_dir": "taobao-ad-x1",
        "files": [
            (
                "TaobaoAd_x1.zip",
                "https://huggingface.co/datasets/reczoo/TaobaoAd_x1/resolve/main/TaobaoAd_x1.zip",
            )
        ],
    },
}


def download_url(url: str, destination: Path) -> None:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    request = urllib.request.Request(url, headers={"User-Agent": "MixFormer-Reproduction/1.0"})
    with urllib.request.urlopen(request, context=ssl_context) as response, open(destination, "wb") as fp:
        shutil.copyfileobj(response, fp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download supported public datasets for MixFormer reproduction.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(DATASET_SPECS.keys()),
    )
    parser.add_argument("--output-dir", type=str, default="data/raw")
    args = parser.parse_args()

    spec = DATASET_SPECS[args.dataset]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    target_dir = output_root / spec["target_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    if spec["kind"] == "zip":
        archive_name, url = spec["files"][0]
        archive_path = output_root / archive_name
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"Dataset already exists at {target_dir}")
            return
        print(f"Downloading {url} -> {archive_path}")
        download_url(url, archive_path)
        print(f"Extracting {archive_path} -> {output_root}")
        with zipfile.ZipFile(archive_path, "r") as zip_fp:
            zip_fp.extractall(output_root)
        print(f"Done. Raw files are available under {target_dir}")
        return

    if spec["kind"] == "zip_flat":
        archive_name, url = spec["files"][0]
        archive_path = output_root / archive_name
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"Dataset already exists at {target_dir}")
            return
        print(f"Downloading {url} -> {archive_path}")
        download_url(url, archive_path)
        print(f"Extracting {archive_path} -> {target_dir}")
        with zipfile.ZipFile(archive_path, "r") as zip_fp:
            zip_fp.extractall(target_dir)
        print(f"Done. Raw files are available under {target_dir}")
        return

    for filename, url in spec["files"]:
        destination = target_dir / filename
        if destination.exists():
            print(f"Skip existing file: {destination}")
            continue
        print(f"Downloading {url} -> {destination}")
        download_url(url, destination)
    print(f"Done. Raw files are available under {target_dir}")


if __name__ == "__main__":
    main()
