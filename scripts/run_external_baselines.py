"""Run five external baselines on the frozen P3/P4 split protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from baselines import BASELINE_REGISTRY, build_baseline  # noqa: E402
from run_ablation import (  # noqa: E402
    make_loader,
    prepare_engine_split,
    regression_metrics,
    set_seed,
    write_json,
)

BASELINES = ("cnn1d", "bilstm", "gru_attention", "transformer", "tcn")
SUBSETS = {"FD001": [101, 202, 303, 404, 505], "FD002": [101, 202, 303]}
LOCKED_WINDOW = 50
EPOCHS = 80
PATIENCE = 20
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4


@torch.no_grad()
def predict(model: nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        predictions.append(model(x).detach().cpu().reshape(-1).numpy())
        targets.append(y.reshape(-1).numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def run_one(subset: str, seed: int, baseline: str, data_dir: Path, output_root: Path, device: torch.device) -> dict:
    run_dir = output_root / subset / f"seed_{seed}" / baseline
    if (run_dir / "metrics.json").is_file():
        return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("test", "RUL"):
        path = data_dir / f"{suffix}_{subset}.txt"
        if path.exists():
            os.chmod(path, 0o444)
    set_seed(seed)
    prepared = prepare_engine_split(
        data_dir=data_dir,
        subset=subset,
        seed=seed,
        validation_fraction=0.2,
        window_size=LOCKED_WINDOW,
        stride=1,
        max_rul=125.0,
        clip_test_rul=True,
        condition_aware=subset in {"FD002", "FD004"},
        n_conditions=6,
    )
    train_loader = make_loader(prepared.train, BATCH, True, 4, device.type == "cuda", seed)
    val_loader = make_loader(prepared.validation, BATCH, False, 4, device.type == "cuda", seed)
    test_loader = make_loader(prepared.test, BATCH, False, 4, device.type == "cuda", seed)
    model = build_baseline(baseline, num_sensors=14, d_model=64, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)
    history = []
    best = float("inf")
    best_epoch = -1
    wait = 0
    best_state = None
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]
        scheduler.step()
        val_pred, val_true = predict(model, val_loader, device)
        val_rmse = float(np.sqrt(np.mean((val_pred - val_true) ** 2)))
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_rmse": val_rmse})
        print(f"[{subset} {baseline} seed={seed}] epoch {epoch:03d} loss={running/max(seen,1):.4f} val_rmse={val_rmse:.4f}", flush=True)
        if val_rmse < best - 1e-6:
            best = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, run_dir / "best_model.pt")
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("No checkpoint")
    model.load_state_dict(best_state)
    test_pred, test_true = predict(model, test_loader, device)
    test_pred = np.clip(test_pred, 0.0, 125.0)
    metrics = regression_metrics(test_true, test_pred, None)
    metrics["params"] = sum(p.numel() for p in model.parameters())
    metrics["best_epoch"] = best_epoch
    metrics["elapsed_seconds"] = time.time() - start
    metrics["subset"] = subset
    metrics["seed"] = seed
    metrics["baseline"] = baseline
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame({
        "unit_id": prepared.test.unit_ids,
        "y_true": test_true,
        "y_pred": test_pred,
        "signed_error": test_pred - test_true,
    }).to_csv(run_dir / "test_predictions.csv", index=False)
    write_json(run_dir / "metrics.json", metrics)
    return metrics


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(np.asarray(a) - np.asarray(b), 0.0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def ensemble_and_stats(output_root: Path, subset: str, seeds: list[int], reference_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = pd.read_csv(reference_csv).sort_values("unit_id").reset_index(drop=True)
    true = ref["y_true"].to_numpy(dtype=float)
    ref_pred = ref["y_pred"].to_numpy(dtype=float)
    ref_abs = np.abs(ref_pred - true)
    ref_nasa = np.where(ref_pred - true < 0, np.expm1(-(ref_pred - true) / 13), np.expm1((ref_pred - true) / 10))
    rows = []
    pred_rows = []
    for baseline in BASELINES:
        frames = []
        for seed in seeds:
            path = output_root / subset / f"seed_{seed}" / baseline / "test_predictions.csv"
            if not path.is_file():
                continue
            frames.append(pd.read_csv(path).sort_values("unit_id").reset_index(drop=True))
        if len(frames) != len(seeds):
            continue
        pred = np.stack([f["y_pred"].to_numpy(dtype=float) for f in frames], axis=0)
        ens = np.clip(pred.mean(axis=0), 0.0, 125.0)
        m = regression_metrics(true, ens, None)
        err = np.abs(ens - true)
        nasa = np.where(ens - true < 0, np.expm1(-(ens - true) / 13), np.expm1((ens - true) / 10))
        rows.append({
            "subset": subset, "baseline": baseline, "seeds": len(frames),
            "rmse": m["rmse"], "nasa_score": m["nasa_score"], "mae": m["mae"],
            "late_ratio": m["late_fraction"], "severe_late_count_gt10": m["severe_late_count_gt10"],
            "params": json.loads((output_root / subset / f"seed_{seeds[0]}" / baseline / "metrics.json").read_text())["params"],
            "p_abs_error_vs_full": safe_wilcoxon(ref_abs, err),
            "p_nasa_vs_full": safe_wilcoxon(ref_nasa, nasa),
        })
        pred_rows.append(pd.DataFrame({"unit_id": ref["unit_id"], "y_true": true, "y_pred": ens, "signed_error": ens - true, "baseline": baseline}))
        if baseline == BASELINES[0]:
            pass
    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "CMAPSS")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "experiments" / "external_baselines")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    os.environ["LD_LIBRARY_PATH"] = "/home/yndx/miniconda3/envs/a40_env/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    protected = [args.data_dir / f"test_{s}.txt" for s in SUBSETS] + [args.data_dir / f"RUL_{s}.txt" for s in SUBSETS]
    for path in protected:
        if path.exists():
            os.chmod(path, 0o444)
    try:
        for subset, seeds in SUBSETS.items():
            for seed in seeds:
                for baseline in BASELINES:
                    run_one(subset, seed, baseline, args.data_dir, args.output_root, device)
        all_rows = []
        all_preds = []
        for subset, seeds in SUBSETS.items():
            ref_path = PROJECT_ROOT / "results" / "ensemble_predictions" / f"{subset}_full_ensemble.csv"
            rows, preds = ensemble_and_stats(args.output_root, subset, seeds, ref_path)
            all_rows.append(rows)
            all_preds.append(preds)
        summary = pd.concat(all_rows, ignore_index=True)
        output = PROJECT_ROOT / "results" / "external_baselines_FD001_FD002.csv"
        summary.to_csv(output, index=False)
        pd.concat(all_preds, ignore_index=True).to_csv(PROJECT_ROOT / "results" / "external_baseline_ensemble_predictions.csv", index=False)
        print(summary.to_string(index=False))
    finally:
        for path in protected:
            if path.exists():
                os.chmod(path, 0o000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
