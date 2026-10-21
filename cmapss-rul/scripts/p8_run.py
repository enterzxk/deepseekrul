"""P8 orchestration: parallel per-(seed, variant) runs with resume."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def build_command(args, subset, variant, seed, run_dir):
    cmd = [
        PY, "-u", str(ROOT / "scripts" / "run_ablation.py"),
        "--subset", subset,
        "--seeds", str(seed),
        "--variants", variant,
        "--run-dir", str(run_dir),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--window-size", str(args.window_size),
        "--dropout", str(args.dropout),
        "--weight-decay", str(args.weight_decay),
        "--asymmetric-weight", str(args.asymmetric_weight),
        "--lambda-mono", str(args.lambda_mono),
        "--late-weight", str(args.late_weight),
        "--identity-prior-alpha", "0.1",
        "--lstm-hidden-size", "70",
        "--clip-test-rul",
        "--overwrite",
        "--num-workers", str(args.num_workers),
        "--device", "cuda",
    ]
    if subset in {"FD002", "FD004"}:
        cmd += ["--condition-aware", "--n-conditions", "6"]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--asymmetric-weight", type=float, default=1.0)
    parser.add_argument("--lambda-mono", type=float, default=0.02)
    parser.add_argument("--late-weight", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--out", default="experiments/p8/final")
    args = parser.parse_args()

    log_dir = ROOT / "logs" / "p8" / args.subset
    log_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = ROOT / args.out / args.subset / f"{variant}_s{seed}"
            done = (run_dir / f"seed_{seed}" / variant / "metrics.json").is_file()
            tasks.append((seed, variant, run_dir, done))

    pending = [t for t in tasks if not t[3]]
    print(f"tasks={len(tasks)} pending={len(pending)} parallel={args.parallel}", flush=True)
    running = []
    failures = []
    while pending or running:
        while pending and len(running) < args.parallel:
            seed, variant, run_dir, _ = pending.pop(0)
            run_dir.mkdir(parents=True, exist_ok=True)
            log = open(log_dir / f"{variant}_s{seed}.log", "a", encoding="utf-8")
            proc = subprocess.Popen(
                build_command(args, args.subset, variant, seed, run_dir),
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            )
            running.append((seed, variant, proc, log))
            print(f"launched {variant} seed={seed}", flush=True)
        time.sleep(5)
        still = []
        for seed, variant, proc, log in running:
            code = proc.poll()
            if code is None:
                still.append((seed, variant, proc, log))
                continue
            log.close()
            print(f"finished {variant} seed={seed} exit={code}", flush=True)
            if code != 0:
                failures.append((variant, seed, code))
        running = still

    if failures:
        print("FAILURES:", failures, flush=True)
        return 1
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
