"""MC-Dropout uncertainty evaluation using frozen FD001 checkpoints."""

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

from model import STG_MambaNet  # noqa: E402
from run_ablation import make_loader, prepare_engine_split, regression_metrics, set_seed  # noqa: E402

SEEDS = [101, 202, 303, 404, 505]
T_MC = 50


def load_full_model(seed: int, device: torch.device) -> torch.nn.Module:
    run_dir = ROOT / "experiments" / "final" / "FD001_locked" / f"seed_{seed}" / f"seed_{seed}" / "full"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model = STG_MambaNet(**config["model_config"]).to(device)
    state = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def mc_predict(model: torch.nn.Module, loader: DataLoader, device: torch.device, samples: int) -> np.ndarray:
    model.train()
    outputs = []
    for _ in range(samples):
        current = []
        for x, _ in loader:
            x = x.to(device)
            current.append(model(x).detach().cpu().reshape(-1).numpy())
        outputs.append(np.concatenate(current))
    return np.stack(outputs, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "uncertainty_frozen_FD001.csv")
    parser.add_argument("--samples", type=int, default=T_MC)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    test_files = [ROOT / "data" / "CMAPSS" / "test_FD001.txt", ROOT / "data" / "CMAPSS" / "RUL_FD001.txt"]
    for path in test_files:
        os.chmod(path, 0o444)
    all_samples = []
    reference_true = None
    unit_ids = None
    try:
        for seed in SEEDS:
            set_seed(seed)
            prepared = prepare_engine_split(
                ROOT / "data" / "CMAPSS", "FD001", seed, 0.2, 50, 1, 125.0, True
            )
            loader = make_loader(prepared.test, 256, False, 4, device.type == "cuda", seed)
            model = load_full_model(seed, device)
            predictions = mc_predict(model, loader, device, args.samples)
            all_samples.append(predictions)
            if reference_true is None:
                reference_true = prepared.test.targets.astype(float).reshape(-1)
                unit_ids = prepared.test.unit_ids
        # (seeds * samples, engines)
        samples = np.concatenate(all_samples, axis=0)
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        lower = np.clip(mean - 1.96 * std, 0.0, 125.0)
        upper = np.clip(mean + 1.96 * std, 0.0, 125.0)
        rows = []
        for i, unit_id in enumerate(unit_ids):
            rows.append({
                "record_type": "engine",
                "unit_id": int(unit_id),
                "y_true": float(reference_true[i]),
                "mean": float(mean[i]),
                "std": float(std[i]),
                "lower95": float(lower[i]),
                "upper95": float(upper[i]),
                "covered95": bool(lower[i] <= reference_true[i] <= upper[i]),
            })
        picp = float(np.mean((lower <= reference_true) & (reference_true <= upper)))
        mpiw = float(np.mean(upper - lower))
        for k in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            safe = np.clip(mean - k * std, 0.0, 125.0)
            m = regression_metrics(reference_true, safe, None)
            rows.append({
                "record_type": "safe_bound",
                "k": k,
                "rmse": m["rmse"],
                "nasa_score": m["nasa_score"],
                "mae": m["mae"],
                "late_fraction": m["late_fraction"],
                "severe_late_count_gt10": m["severe_late_count_gt10"],
                "PICP95": picp,
                "MPIW95": mpiw,
            })
        pd.DataFrame(rows).to_csv(args.output, index=False)
        np.savez_compressed(
            ROOT / "results" / "uncertainty_frozen_FD001.npz",
            mean=mean,
            std=std,
            lower95=lower,
            upper95=upper,
            y_true=reference_true,
            unit_ids=unit_ids,
        )
        print(json.dumps({"PICP95": picp, "MPIW95": mpiw, "n_samples": len(samples)}, indent=2))
    finally:
        if os.environ.get("KEEP_TEST_UNSEALED") != "1":
            for path in test_files:
                os.chmod(path, 0o000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
