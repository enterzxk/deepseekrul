"""Validation-only training for P1/P2.

This script deliberately never reads test_FDxxx.txt or RUL_FDxxx.txt.  It
builds train/validation windows from training engines only, uses training-only
Min-Max statistics, and selects checkpoints by validation RMSE.

Example:
    python scripts/tune_validation.py --tag p1_w30_d02_wd1e4 \
      --window-size 30 --dropout 0.2 --weight-decay 1e-4 \
      --seeds 42 67 123 --epochs 80 --patience 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import (  # noqa: E402
    CMAPSSDataset,
    SELECTED_SENSOR_COLUMNS,
    SensorMinMaxScaler,
    _build_train_windows,
    _read_whitespace_table,
    compute_nasa_score,
    compute_rmse,
)
from loss import PhysicsGuidedLoss  # noqa: E402
from model import STG_MambaNet  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_engines(unit_ids: np.ndarray, validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    units = np.sort(np.asarray(unit_ids, dtype=np.int64))
    n_validation = int(round(units.size * validation_fraction))
    n_validation = min(max(n_validation, 1), units.size - 1)
    rng = np.random.default_rng(seed + 20261020)
    shuffled = rng.permutation(units)
    validation_units = sorted(int(x) for x in shuffled[:n_validation])
    train_units = sorted(int(x) for x in shuffled[n_validation:])
    return train_units, validation_units


def prepare_train_validation(
    data_dir: Path,
    subset: str,
    seed: int,
    validation_fraction: float,
    window_size: int,
    stride: int,
    max_rul: float,
) -> tuple[CMAPSSDataset, CMAPSSDataset, list[int], list[int]]:
    raw_train = _read_whitespace_table(data_dir / f"train_{subset}.txt")
    train_units, validation_units = split_engines(
        raw_train["unit"].unique(), validation_fraction, seed
    )
    train_frame = (
        raw_train.loc[raw_train["unit"].isin(set(train_units))]
        .sort_values(["unit", "cycle"])
        .reset_index(drop=True)
    )
    validation_frame = (
        raw_train.loc[raw_train["unit"].isin(set(validation_units))]
        .sort_values(["unit", "cycle"])
        .reset_index(drop=True)
    )

    columns = list(SELECTED_SENSOR_COLUMNS)
    scaler = SensorMinMaxScaler(columns)
    train_sensors = scaler.fit_transform(
        train_frame.loc[:, columns].to_numpy(dtype=np.float64)
    )
    validation_sensors = scaler.transform(
        validation_frame.loc[:, columns].to_numpy(dtype=np.float64)
    )

    train_windows = _build_train_windows(
        train_frame, train_sensors, window_size, stride, max_rul
    )
    validation_windows = _build_train_windows(
        validation_frame, validation_sensors, window_size, stride, max_rul
    )
    train_dataset = CMAPSSDataset(
        train_windows[0], train_windows[1], train_windows[2], train_windows[3]
    )
    validation_dataset = CMAPSSDataset(
        validation_windows[0],
        validation_windows[1],
        validation_windows[2],
        validation_windows[3],
    )
    return train_dataset, validation_dataset, train_units, validation_units


def make_loader(dataset: CMAPSSDataset, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def predict_validation(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    trajectories = []
    for x, y in loader:
        prediction, trajectory = model(x.to(device), return_trajectory=True)
        predictions.append(prediction.cpu().numpy().reshape(-1))
        targets.append(y.numpy().reshape(-1))
        trajectories.append(trajectory.cpu().numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(trajectories),
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray, trajectory: np.ndarray) -> dict:
    error = y_pred - y_true
    increments = np.diff(trajectory, axis=1)
    return {
        "val_rmse": float(compute_rmse(y_true, y_pred)),
        "val_nasa": float(compute_nasa_score(y_true, y_pred)),
        "val_mae": float(np.mean(np.abs(error))),
        "val_mean_bias": float(np.mean(error)),
        "val_late_fraction": float(np.mean(error > 0.0)),
        "val_severe_late_count_gt10": int(np.sum(error > 10.0)),
        "val_mono_violation_rate": float(np.mean(increments > 0.0)),
        "val_n": int(y_true.size),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "CMAPSS")
    p.add_argument("--subset", default="FD001")
    p.add_argument("--tag", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 67, 123])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-rul", type=float, default=125.0)
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--graph-embed-dim", type=int, default=16)
    p.add_argument("--num-gcn-layers", type=int, default=2)
    p.add_argument("--state-dim", type=int, default=16)
    p.add_argument("--kernels", nargs="+", type=int, default=[3, 5, 7])
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--graph-mode", choices=("learnable", "identity", "random_fixed", "hollow"), default="learnable")
    p.add_argument("--identity-prior-alpha", type=float, default=0.1)
    p.add_argument("--hollow-alpha", type=float, default=0.35)
    p.add_argument("--monotonic-head", action="store_true")
    p.add_argument("--use-delta-features", action="store_true")
    p.add_argument("--condition-film", action="store_true")
    p.add_argument("--temporal-mode", choices=("ssm", "lstm", "bilstm_attn"), default="ssm")
    p.add_argument("--lstm-hidden-size", type=int, default=70)
    p.add_argument("--lstm-num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--loss-mode", choices=("mse", "asym", "mono", "full"), default="full")
    p.add_argument("--mse-weight", type=float, default=1.0)
    p.add_argument("--asymmetric-weight", type=float, default=0.5)
    p.add_argument("--late-weight", type=float, default=2.0)
    p.add_argument("--lambda-mono", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "experiments" / "validation_tuning")
    p.add_argument("--val-only", action="store_true", default=True, help="Validation-only; never reads test files.")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    run_root = args.output_root / args.tag
    run_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in args.seeds:
        existing = run_root / f"seed_{seed}" / "validation_metrics.json"
        if args.resume and existing.is_file():
            print(f"[{args.tag}] seed={seed} already complete; skip", flush=True)
            all_results.append(json.loads(existing.read_text(encoding="utf-8")))
            continue
        set_seed(seed)
        train_set, val_set, train_units, validation_units = prepare_train_validation(
            args.data_dir,
            args.subset,
            seed,
            args.validation_fraction,
            args.window_size,
            args.stride,
            args.max_rul,
        )
        train_loader = make_loader(train_set, args.batch_size, True, seed, args.num_workers)
        val_loader = make_loader(val_set, args.batch_size, False, seed, args.num_workers)

        model = STG_MambaNet(
            num_sensors=14,
            window_size=args.window_size,
            d_model=args.d_model,
            graph_embed_dim=args.graph_embed_dim,
            num_gcn_layers=args.num_gcn_layers,
            state_dim=args.state_dim,
            kernels=args.kernels,
            num_heads=args.num_heads,
            dropout=args.dropout,
            graph_mode=args.graph_mode,
            identity_prior_alpha=args.identity_prior_alpha,
            hollow_alpha=args.hollow_alpha,
            monotonic_head=args.monotonic_head,
            use_delta_features=args.use_delta_features,
            condition_film=args.condition_film,
            temporal_mode=args.temporal_mode,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
        ).to(device)

        criterion = PhysicsGuidedLoss(
            mse_weight=args.mse_weight,
            asymmetric_weight=args.asymmetric_weight,
            late_weight=args.late_weight,
            use_physics=args.loss_mode != "mse",
            use_asym=args.loss_mode in {"asym", "full"},
            use_mono=args.loss_mode in {"mono", "full"},
            lambda_mono=args.lambda_mono,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

        history = []
        best_rmse = float("inf")
        best_epoch = -1
        best_state = None
        bad_epochs = 0
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            seen = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction, trajectory = model(x, return_trajectory=True)
                loss = criterion(prediction, y, y_pred_traj=trajectory)
                loss.backward()
                if args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                running += float(loss.detach()) * x.shape[0]
                seen += x.shape[0]
            scheduler.step()

            val_pred, val_true, val_traj = predict_validation(model, val_loader, device)
            val_metrics = metrics(val_true, val_pred, val_traj)
            row = {
                "epoch": epoch,
                "train_loss": running / max(seen, 1),
                **val_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(
                f"[{args.tag} seed={seed}] epoch {epoch:03d} "
                f"loss={row['train_loss']:.3f} val_rmse={row['val_rmse']:.3f} "
                f"val_nasa={row['val_nasa']:.2f}",
                flush=True,
            )
            if val_metrics["val_rmse"] < best_rmse - 1e-4:
                best_rmse = val_metrics["val_rmse"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"[{args.tag} seed={seed}] early stop at epoch {epoch}", flush=True)
                    break

        if best_state is None:
            raise RuntimeError("No finite validation checkpoint.")
        model.load_state_dict(best_state)
        val_pred, val_true, val_traj = predict_validation(model, val_loader, device)
        best_metrics = metrics(val_true, val_pred, val_traj)
        seed_dir = run_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with (seed_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        result = {
            "tag": args.tag,
            "seed": seed,
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "best_validation_rmse": best_rmse,
            "best_metrics": best_metrics,
            "num_parameters": model.num_parameters,
            "train_units": train_units,
            "validation_units": validation_units,
            "config": vars(args),
            "elapsed_seconds": time.time() - start,
        }
        (seed_dir / "validation_metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        all_results.append(result)

    (run_root / "all_seeds.json").write_text(
        json.dumps({"results": all_results}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Completed {args.tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
