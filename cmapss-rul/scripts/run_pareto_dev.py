"""Validation-only Pareto scan for StateAwarePhysicsLoss gamma."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--gammas", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0, 4.0])
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=40.0)
    parser.add_argument("--asymmetric-weight", type=float, default=0.01)
    parser.add_argument("--lambda-selection", type=float, default=0.5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    rows = []
    for gamma in args.gammas:
        run_dir = args.output_dir / f"gamma_{gamma:g}"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_dev_fold.py"),
            "--fold",
            str(args.fold),
            "--run-dir",
            str(run_dir),
        ]
        overrides = {
            **base,
            "gamma": gamma,
            "eta": args.eta,
            "tau": args.tau,
            "asymmetric_weight": args.asymmetric_weight,
            "base_weight": 1.0,
            "selection_metric": "composite",
            "lambda_selection": args.lambda_selection,
        }
        for key, value in overrides.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            elif isinstance(value, list):
                command.append(flag)
                command.extend(str(item) for item in value)
            else:
                command.extend([flag, str(value)])
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        vm = metrics["validation_metrics"]
        rows.append(
            {
                "gamma": gamma,
                "eta": args.eta,
                "tau": args.tau,
                "asymmetric_weight": args.asymmetric_weight,
                "fold": args.fold,
                "best_epoch": metrics["best_epoch"],
                "rmse": vm["rmse"],
                "nasa_score": vm["nasa_score"],
                "mae": vm["mae"],
                "late_ratio": vm["late_ratio"],
                "severe_late_ratio": vm["severe_late_ratio"],
                "mean_bias": vm["mean_bias"],
                "composite": vm["composite"],
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "pareto_tradeoff.csv", index=False)
    # Dominance is computed on validation only.
    metrics = ["rmse", "nasa_score", "late_ratio", "severe_late_ratio"]
    values = frame[metrics].to_numpy(dtype=float)
    pareto = []
    for i in range(len(frame)):
        dominated = False
        for j in range(len(frame)):
            if i == j:
                continue
            if np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
                dominated = True
                break
        pareto.append(not dominated)
    frame["is_pareto"] = pareto
    frame.to_csv(args.output_dir / "pareto_tradeoff.csv", index=False)
    print(frame.to_string(index=False), flush=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"rows": rows, "pareto": pareto}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
