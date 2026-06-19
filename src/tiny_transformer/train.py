"""Command-line entry point for deterministic training and experiment sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from torch import nn

from tiny_transformer.data import DataLoaders, create_dataloaders
from tiny_transformer.evaluate import evaluate
from tiny_transformer.models import build_model, count_trainable_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float,
) -> float:
    model.train()
    loss_sum = 0.0
    token_count = 0
    for source, target, valid_mask in loader:
        source, target, valid_mask = (
            source.to(device),
            target.to(device),
            valid_mask.to(device),
        )
        optimizer.zero_grad(set_to_none=True)
        target_input = torch.cat((torch.ones_like(target[:, :1]), target[:, :-1]), dim=1)
        logits = model(source, valid_mask, target_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        valid_tokens = valid_mask.sum().item()
        loss_sum += loss.item() * valid_tokens
        token_count += valid_tokens
    return loss_sum / max(token_count, 1)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_summaries(output_dir: Path) -> list[dict]:
    summaries = []
    for path in sorted(output_dir.glob("*/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def run_experiment(
    config: dict,
    model_name: str,
    num_heads: int,
    seed: int,
    device: torch.device,
) -> dict:
    set_seed(seed)
    data_config = {**config["data"], "batch_size": config["training"]["batch_size"]}
    loaders: DataLoaders = create_dataloaders(data_config, seed)
    model = build_model(
        model_name,
        config["model"],
        vocabulary_size=int(config["data"]["vocabulary_size"]),
        num_heads=num_heads,
    ).to(device)

    training_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    heads_label = str(num_heads) if model_name == "transformer" else "na"
    run_name = f"{model_name}_heads-{heads_label}_seed-{seed}"
    run_dir = Path(config["experiment"]["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_score = float("-inf")
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, int(training_config["epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_loss = train_epoch(
            model,
            loaders.train,
            optimizer,
            criterion,
            device,
            float(training_config["gradient_clip"]),
        )
        validation = evaluate(model, loaders.validation, device, autoregressive=False)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "validation_token_accuracy": validation["token_accuracy"],
            "validation_sequence_accuracy": validation["sequence_accuracy"],
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        write_csv(run_dir / "history.csv", history)
        score = -validation["loss"]
        if score > best_score:
            best_score = score
            stale_epochs = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            stale_epochs += 1
        print(
            f"{run_name} epoch={epoch:02d} loss={train_loss:.4f} "
            f"val_token={validation['token_accuracy']:.4f} "
            f"val_sequence={validation['sequence_accuracy']:.4f}"
        )
        if stale_epochs >= int(training_config["early_stopping_patience"]):
            break

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    validation = evaluate(model, loaders.validation, device)
    test = evaluate(model, loaders.test, device)
    generalization = evaluate(model, loaders.generalization, device)
    summary = {
        "run": run_name,
        "model": model_name,
        "num_heads": num_heads if model_name == "transformer" else None,
        "seed": seed,
        "parameters": count_trainable_parameters(model),
        "epochs_completed": len(history),
        "training_seconds": time.perf_counter() - started,
        "validation_token_accuracy": validation["token_accuracy"],
        "validation_sequence_accuracy": validation["sequence_accuracy"],
        "test_token_accuracy": test["token_accuracy"],
        "test_sequence_accuracy": test["sequence_accuracy"],
        "generalization_token_accuracy": generalization["token_accuracy"],
        "generalization_sequence_accuracy": generalization["sequence_accuracy"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_list(value: str | None, fallback: list, cast) -> list:
    return fallback if value is None else [cast(item.strip()) for item in value.split(",")]


def smoke_config(config: dict) -> dict:
    result = deepcopy(config)
    result["experiment"]["output_dir"] = "artifacts/smoke"
    result["data"].update(
        {"train_size": 128, "validation_size": 64, "test_size": 64, "train_length": [3, 8]}
    )
    result["data"]["generalization_length"] = [9, 12]
    result["model"].update({"d_model": 32, "num_layers": 1, "dim_feedforward": 64})
    result["training"].update({"batch_size": 32, "epochs": 2, "early_stopping_patience": 2})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", help="Comma-separated subset: lstm,transformer")
    parser.add_argument("--heads", help="Comma-separated Transformer head counts")
    parser.add_argument("--seeds", help="Comma-separated random seeds")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip runs with an existing summary")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke_test:
        config = smoke_config(config)
    models = parse_list(args.models, list(config["model"]["types"]), str)
    heads = parse_list(args.heads, list(config["model"]["num_heads"]), int)
    seeds = parse_list(args.seeds, list(config["experiment"]["seeds"]), int)
    device = select_device(str(config["training"]["device"]))
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "config": config,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summaries: list[dict] = collect_summaries(output_dir) if args.resume else []
    for model_name in models:
        model_heads = heads if model_name == "transformer" else [heads[0]]
        for num_heads in model_heads:
            for seed in seeds:
                heads_label = str(num_heads) if model_name == "transformer" else "na"
                run_dir = output_dir / f"{model_name}_heads-{heads_label}_seed-{seed}"
                if args.resume and (run_dir / "summary.json").exists():
                    print(f"Skipping completed run: {run_dir.name}")
                    continue
                summaries.append(run_experiment(config, model_name, num_heads, seed, device))
                write_csv(output_dir / "runs.csv", summaries)
    summaries = collect_summaries(output_dir)
    write_csv(output_dir / "runs.csv", summaries)
    write_csv(Path(config["experiment"]["results_file"]), summaries)
    (output_dir / "runs.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
