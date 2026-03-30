#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mixformer.data.dataset import BatchBuilder, MovieLensMixFormerDataset
from mixformer.data.preprocess import load_bundle
from mixformer.models.mixformer import MixFormerModel
from mixformer.reporting import count_trainable_params_m, estimate_flops_per_batch_g, measure_average_latency_ms
from mixformer.trainer import evaluate, train_one_epoch
from mixformer.utils import ensure_dir, load_config, resolve_device, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MixFormer on a processed public dataset.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-negatives", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        config["training"]["eval_batch_size"] = args.eval_batch_size
    if args.eval_negatives is not None:
        config["training"]["eval_negatives"] = args.eval_negatives
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.output_dir is not None:
        config["training"]["output_dir"] = args.output_dir

    set_seed(config["training"]["seed"])

    bundle = load_bundle(config["dataset"]["processed_path"])
    device = resolve_device(config["training"]["device"])
    output_dir = ensure_dir(config["training"]["output_dir"])

    train_samples = bundle["splits"]["train"][: args.limit_train] if args.limit_train else bundle["splits"]["train"]
    val_samples = bundle["splits"]["val"][: args.limit_val] if args.limit_val else bundle["splits"]["val"]
    test_samples = bundle["splits"]["test"][: args.limit_test] if args.limit_test else bundle["splits"]["test"]

    train_dataset = MovieLensMixFormerDataset(train_samples)
    val_dataset = MovieLensMixFormerDataset(val_samples)
    test_dataset = MovieLensMixFormerDataset(test_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=lambda x: x,
    )
    eval_loader_kwargs = {
        "batch_size": config["training"]["eval_batch_size"],
        "shuffle": False,
        "num_workers": config["training"]["num_workers"],
        "collate_fn": lambda x: x,
    }
    val_loader = DataLoader(val_dataset, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, **eval_loader_kwargs)

    batch_builder = BatchBuilder(
        user_features=bundle["user_features"],
        item_features=bundle["item_features"],
        all_item_ids=bundle.get("candidate_item_ids", list(bundle["item_features"].keys())),
        user_seen_items=bundle["user_seen_items"],
        max_seq_len=config["model"]["max_seq_len"],
        max_genres_per_item=bundle["meta"]["max_genres_per_item"],
        seed=config["training"]["seed"],
    )

    model = MixFormerModel(bundle["meta"], config["model"]).to(device)
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_metric = float("-inf")
    best_path = output_dir / "best_model.pt"
    history = []
    top_k = config["training"]["top_k"]

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            batch_builder=batch_builder,
            optimizer=optimizer,
            device=device,
            negative_ratio=config["training"]["negative_ratio"],
            grad_clip=config["training"]["grad_clip"],
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            batch_builder=batch_builder,
            device=device,
            num_negatives=config["training"]["eval_negatives"],
            top_k=top_k,
        )
        epoch_summary = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(epoch_summary)
        print(epoch_summary)

        key_metric = val_metrics[f"ndcg@{top_k}"]
        if key_metric > best_metric:
            best_metric = key_metric
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "meta": bundle["meta"],
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        batch_builder=batch_builder,
        device=device,
        num_negatives=config["training"]["eval_negatives"],
        top_k=top_k,
    )

    efficiency_metrics = {}
    try:
        reference_samples = test_samples[: min(8, len(test_samples))]
        if reference_samples:
            reference_batch, _ = batch_builder.build_eval_batch(
                reference_samples,
                num_negatives=min(config["training"]["eval_negatives"], 20),
            )
            reference_batch = {key: value.to(device) for key, value in reference_batch.items()}
            efficiency_metrics = {
                "params_m": count_trainable_params_m(model),
                "approx_gflops_per_batch": estimate_flops_per_batch_g(config["model"], reference_batch),
                "avg_latency_ms": measure_average_latency_ms(model, reference_batch, device=device),
            }
    except Exception as exc:  # noqa: BLE001
        efficiency_metrics = {"metric_error": str(exc)}

    result = {
        "best_val_ndcg": best_metric,
        "test_metrics": test_metrics,
        "efficiency_metrics": efficiency_metrics,
        "history": history,
        "device": str(device),
    }
    save_json(output_dir / "metrics.json", result)
    print("Final test metrics:", test_metrics)
    if efficiency_metrics:
        print("Efficiency metrics:", efficiency_metrics)
    print(f"Saved best model to {best_path}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
