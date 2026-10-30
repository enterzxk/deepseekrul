"""Noise and sensor-dropout stress test for frozen FD001 models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from baselines import build_baseline  # noqa: E402
from model import STG_MambaNet  # noqa: E402
from run_ablation import make_loader, prepare_engine_split, regression_metrics, set_seed  # noqa: E402

SEEDS = [101, 202, 303, 404, 505]
MODEL_VARIANTS = ("full", "wo_agcn", "identity_graph")
CONDITIONS = [
    ("clean", 0.0, 0),
    ("noise_0.05", 0.05, 0),
    ("noise_0.10", 0.10, 0),
    ("noise_0.15", 0.15, 0),
    ("noise_0.20", 0.20, 0),
    ("dropout_1", 0.0, 1),
    ("dropout_2", 0.0, 2),
]


def load_locked_model(seed: int, variant: str, device: torch.device) -> torch.nn.Module:
    path = ROOT / "experiments" / "final" / "FD001_locked" / f"seed_{seed}" / f"seed_{seed}" / variant
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model = STG_MambaNet(**config["model_config"]).to(device)
    model.load_state_dict(torch.load(path / "best_model.pt", map_location=device, weights_only=True))
    return model


def load_bilstm(seed: int, device: torch.device) -> torch.nn.Module | None:
    path = ROOT / "experiments" / "external_baselines" / f"seed_{seed}" / "bilstm"
    if not (path / "best_model.pt").is_file():
        return None
    model = build_baseline("bilstm", num_sensors=14, d_model=64, dropout=0.2).to(device)
    model.load_state_dict(torch.load(path / "best_model.pt", map_location=device, weights_only=True))
    return model


@torch.no_grad()
def perturbed_predict(model, loader, device, noise: float, dropout_channels: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    rng = np.random.default_rng(seed + 991)
    preds, truths = [], []
    for x, y in loader:
        x = x.to(device)
        if noise > 0:
            noise_array = rng.standard_normal(x.shape).astype(np.float32)
            x = x + noise * torch.from_numpy(noise_array).to(device)
        if dropout_channels > 0:
            indices = rng.choice(14, size=dropout_channels, replace=False)
            x = x.clone()
            x[:, :, torch.as_tensor(indices, device=device)] = 0.0
        preds.append(model(x).detach().cpu().reshape(-1).numpy())
        truths.append(y.numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(truths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "robustness_stress_FD001.csv")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    test_files = [ROOT / "data" / "CMAPSS" / "test_FD001.txt", ROOT / "data" / "CMAPSS" / "RUL_FD001.txt"]
    for path in test_files:
        os.chmod(path, 0o444)
    rows = []
    try:
        for condition, noise, dropout_channels in CONDITIONS:
            for model_name in (*MODEL_VARIANTS, "bilstm"):
                predictions = []
                truths = []
                for seed in SEEDS:
                    set_seed(seed)
                    prepared = prepare_engine_split(
                        ROOT / "data" / "CMAPSS", "FD001", seed, 0.2, 50, 1, 125.0, True
                    )
                    loader = make_loader(prepared.test, 256, False, 4, device.type == "cuda", seed)
                    if model_name == "bilstm":
                        model = load_bilstm(seed, device)
                    else:
                        model = load_locked_model(seed, model_name, device)
                    if model is None:
                        continue
                    pred, true = perturbed_predict(model, loader, device, noise, dropout_channels, seed)
                    predictions.append(np.clip(pred, 0.0, 125.0))
                    truths.append(true)
                if not predictions:
                    continue
                ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
                metrics = regression_metrics(truths[0], ensemble, None)
                rows.append({
                    "condition": condition,
                    "noise_sigma": noise,
                    "dropped_sensors": dropout_channels,
                    "model": model_name,
                    "rmse": metrics["rmse"],
                    "nasa_score": metrics["nasa_score"],
                    "mae": metrics["mae"],
                    "late_fraction": metrics["late_fraction"],
                    "severe_late_count_gt10": metrics["severe_late_count_gt10"],
                })
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(pd.DataFrame(rows).to_string(index=False))
    finally:
        if os.environ.get("KEEP_TEST_UNSEALED") != "1":
            for path in test_files:
                os.chmod(path, 0o000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
