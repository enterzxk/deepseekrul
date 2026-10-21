"""Blind development training for one engine-level C-MAPSS fold.

This script never reads official test or RUL files. It refuses to run if those
files are readable, enforcing the Stage 1 blind-test-freeze gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import (  # noqa: E402
    compute_degradation_weights,
    prepare_development_fold,
)
from loss import (  # noqa: E402
    StateAwarePhysicsLoss,
    composite_validation_score,
    risk_metrics,
)
from model import Lite_STG_Mamba  # noqa: E402
from baselines import BASELINE_REGISTRY, build_baseline  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def assert_blind_freeze(data_dir: Path, subset: str) -> None:
    blocked = [data_dir / f"test_{subset}.txt", data_dir / f"RUL_{subset}.txt"]
    for path in blocked:
        if not path.is_file():
            raise FileNotFoundError(f"Required freeze target missing: {path}")
        os.chmod(path, 0o000)
    readable = [str(path) for path in blocked if os.access(path, os.R_OK)]
    if readable:
        raise RuntimeError(
            "Failed to enforce blind test freeze for: " + str(readable)
        )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scale: float,
    lambda_selection: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        output = model(x)
        predictions.append(output.detach().cpu().reshape(-1).numpy())
        targets.append(y.detach().cpu().reshape(-1).numpy())
    pred_norm = np.concatenate(predictions)
    true_norm = np.concatenate(targets)
    pred = np.clip(pred_norm * target_scale, 0.0, target_scale)
    true = true_norm * target_scale
    metrics = risk_metrics(true, pred)
    metrics["composite"] = composite_validation_score(
        metrics["rmse"], metrics["nasa_score"], lambda_selection
    )
    return metrics, true, pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "CMAPSS")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument(
        "--model",
        choices=("lite", *BASELINE_REGISTRY.keys()),
        default="lite",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--scaler", choices=("minmax", "standard"), default="minmax")
    parser.add_argument("--condition-aware", action="store_true")
    parser.add_argument("--n-conditions", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--graph-embed-dim", type=int, default=16)
    parser.add_argument("--num-gcn-layers", type=int, default=2)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--kernels", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--output-activation", choices=("linear", "sigmoid"), default="linear")
    parser.add_argument("--bidirectional-ssm", choices=("on", "off"), default="on")
    parser.add_argument("--use-graph", choices=("on", "off"), default="on")
    parser.add_argument("--use-delta", choices=("on", "off"), default="on")
    parser.add_argument("--temporal-mode", choices=("ssm", "lstm"), default="ssm")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=40.0)
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--asymmetric-weight", type=float, default=1.0)
    parser.add_argument("--smoothl1-beta", type=float, default=5.0)
    parser.add_argument("--selection-metric", choices=("composite", "rmse"), default="composite")
    parser.add_argument("--lambda-selection", type=float, default=0.5)
    parser.add_argument("--degradation-weighting", choices=("on", "off"), default="on")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assert_blind_freeze(args.data_dir, args.subset)
    if args.run_dir.exists():
        raise FileExistsError(f"Run directory exists: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    set_seed(args.model_seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    fold = prepare_development_fold(
        data_dir=args.data_dir,
        subset=args.subset,
        fold_index=args.fold,
        n_splits=args.n_splits,
        seed=args.split_seed,
        scaler_type=args.scaler,
        window_size=args.window_size,
        stride=args.stride,
        target_scale=125.0,
        condition_aware=args.condition_aware,
        n_conditions=args.n_conditions,
    )
    train_dataset = fold.train
    validation_dataset = fold.validation

    train_weights = compute_degradation_weights(train_dataset.targets, num_bins=5)
    generator = torch.Generator().manual_seed(args.model_seed)
    sampler = (
        WeightedRandomSampler(
            torch.as_tensor(train_weights, dtype=torch.double),
            num_samples=len(train_weights),
            replacement=True,
            generator=generator,
        )
        if args.degradation_weighting == "on"
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator if sampler is None else None,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    if args.model == "lite":
        model = Lite_STG_Mamba(
            num_sensors=14,
            window_size=args.window_size,
            d_model=args.d_model,
            graph_embed_dim=args.graph_embed_dim,
            num_gcn_layers=args.num_gcn_layers,
            state_dim=args.state_dim,
            kernels=args.kernels,
            dropout=args.dropout,
            target_scale=125.0,
            output_activation=args.output_activation,
            bidirectional_ssm=args.bidirectional_ssm == "on",
            use_graph=args.use_graph == "on",
            use_delta=args.use_delta == "on",
            temporal_mode=args.temporal_mode,
        ).to(device)
    else:
        model = build_baseline(
            args.model,
            num_sensors=14,
            d_model=args.d_model,
            dropout=args.dropout,
        ).to(device)
    criterion = StateAwarePhysicsLoss(
        target_scale=125.0,
        gamma=args.gamma,
        eta=args.eta,
        tau=args.tau,
        base_weight=args.base_weight,
        asymmetric_weight=args.asymmetric_weight,
        smoothl1_beta=args.smoothl1_beta,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    scaler_state = (
        fold.scaler.state_dict()
        if hasattr(fold.scaler, "state_dict")
        else {"type": "minmax", "feature_range": fold.scaler.feature_range()}
    )
    config = {
        **vars(args),
        "data_dir": str(args.data_dir),
        "run_dir": str(args.run_dir),
        "train_units": fold.train_units,
        "validation_units": fold.validation_units,
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "model": args.model,
        "num_parameters": (
            model.num_parameters
            if hasattr(model, "num_parameters")
            else sum(parameter.numel() for parameter in model.parameters())
        ),
        "target_scale": 125.0,
        "test_files_accessed": False,
    }
    write_json(args.run_dir / "config.json", config)
    write_json(args.run_dir / "scaler_state.json", scaler_state)

    history: list[dict] = []
    best_score = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]
        scheduler.step()

        metrics, _, _ = evaluate(
            model,
            validation_loader,
            device,
            125.0,
            args.lambda_selection,
        )
        selection_value = (
            metrics["composite"]
            if args.selection_metric == "composite"
            else metrics["rmse"]
        )
        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_rmse": metrics["rmse"],
            "val_nasa_score": metrics["nasa_score"],
            "val_mae": metrics["mae"],
            "val_late_ratio": metrics["late_ratio"],
            "val_severe_late_ratio": metrics["severe_late_ratio"],
            "val_composite": metrics["composite"],
            "selection_value": selection_value,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(row)
        print(
            f"[{args.subset} fold={args.fold} seed={args.model_seed}] "
            f"epoch {epoch:03d} loss={row['train_loss']:.5f} "
            f"val_rmse={row['val_rmse']:.4f} val_nasa={row['val_nasa_score']:.3f} "
            f"late={row['val_late_ratio']:.3f} comp={row['val_composite']:.4f}",
            flush=True,
        )
        if selection_value < best_score - 1e-6:
            best_score = selection_value
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            torch.save(best_state, args.run_dir / "best_model.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stop at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("No checkpoint selected.")
    model.load_state_dict(best_state)
    final_metrics, final_true, final_pred = evaluate(
        model,
        validation_loader,
        device,
        125.0,
        args.lambda_selection,
    )
    with (args.run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    pd.DataFrame(
        {
            "unit_id": validation_dataset.unit_ids,
            "end_cycle": validation_dataset.end_cycles,
            "y_true": final_true,
            "y_pred": final_pred,
            "signed_error": final_pred - final_true,
        }
    ).to_csv(args.run_dir / "validation_predictions.csv", index=False)
    result = {
        "subset": args.subset,
        "fold": args.fold,
        "model_seed": args.model_seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_selection_value": best_score,
        "num_parameters": (
            model.num_parameters
            if hasattr(model, "num_parameters")
            else sum(parameter.numel() for parameter in model.parameters())
        ),
        "validation_metrics": final_metrics,
        "elapsed_seconds": time.time() - start,
        "test_files_accessed": False,
    }
    write_json(args.run_dir / "metrics.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
