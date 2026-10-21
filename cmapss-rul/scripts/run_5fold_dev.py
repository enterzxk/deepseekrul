"""Run a blind development configuration over multiple engine folds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output_root.exists():
        raise FileExistsError(f"Output root exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    results = []
    for fold in args.folds:
        run_dir = args.output_root / f"fold_{fold}"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_dev_fold.py"),
            "--fold",
            str(fold),
            "--run-dir",
            str(run_dir),
        ]
        for key, value in config.items():
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
        results.append(metrics)

    rmses = np.asarray([r["validation_metrics"]["rmse"] for r in results])
    nasa = np.asarray([r["validation_metrics"]["nasa_score"] for r in results])
    maes = np.asarray([r["validation_metrics"]["mae"] for r in results])
    summary = {
        "folds": args.folds,
        "n_folds": len(results),
        "rmse_mean": float(rmses.mean()),
        "rmse_std": float(rmses.std(ddof=0)),
        "nasa_mean": float(nasa.mean()),
        "nasa_std": float(nasa.std(ddof=0)),
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std(ddof=0)),
        "per_fold_rmse": rmses.tolist(),
        "per_fold_nasa": nasa.tolist(),
        "test_files_accessed": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "fold": args.folds,
            "rmse": rmses,
            "nasa_score": nasa,
            "mae": maes,
        }
    ).to_csv(args.output_root / "per_fold_metrics.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
