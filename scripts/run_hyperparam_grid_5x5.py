"""Validation-only 5x5 gamma/tau grid for StateAwarePhysicsLoss."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--gammas", nargs="+", type=float, default=[1.0, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--taus", nargs="+", type=float, default=[20.0, 30.0, 40.0, 50.0, 60.0])
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--asymmetric-weight", type=float, default=0.01)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    rows = []
    for gamma in args.gammas:
        for tau in args.taus:
            name = f"g{gamma:g}_tau{tau:g}"
            run_dir = args.output_dir / name
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train_dev_fold.py"),
                "--fold", str(args.fold),
                "--run-dir", str(run_dir),
            ]
            overrides = {
                **base,
                "gamma": gamma,
                "tau": tau,
                "eta": args.eta,
                "asymmetric_weight": args.asymmetric_weight,
                "base_weight": 1.0,
                "selection_metric": "composite",
                "lambda_selection": 0.5,
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
            try:
                subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            except subprocess.CalledProcessError as error:
                print("FAILED", name, error, flush=True)
                rows.append({"gamma": gamma, "tau": tau, "status": "failed"})
                continue
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            vm = metrics["validation_metrics"]
            rows.append(
                {
                    "gamma": gamma,
                    "tau": tau,
                    "status": "ok",
                    "best_epoch": metrics["best_epoch"],
                    "rmse": vm["rmse"],
                    "nasa_score": vm["nasa_score"],
                    "mae": vm["mae"],
                    "late_ratio": vm["late_ratio"],
                    "severe_late_ratio": vm["severe_late_ratio"],
                    "composite": vm["composite"],
                }
            )
            pd.DataFrame(rows).to_csv(args.output_dir / "hyperparam_grid_5x5.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "hyperparam_grid_5x5.csv", index=False)
    print(frame.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
