"""Run the FD002 multi-condition external-validity experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_ablation.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    locked = json.loads((ROOT / "results" / "locked_hparams.json").read_text(encoding="utf-8"))
    full_loss = locked["full_loss"]
    test_files = [
        ROOT / "data" / "CMAPSS" / "test_FD002.txt",
        ROOT / "data" / "CMAPSS" / "RUL_FD002.txt",
    ]
    for path in test_files:
        os.chmod(path, 0o644)
    log_dir = ROOT / "logs" / "fd002"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_root = ROOT / "experiments" / "final" / "FD002_3seeds"
    run_root.mkdir(parents=True, exist_ok=True)

    pending = []
    for seed in (101, 202, 303):
        run_dir = run_root / f"seed_{seed}"
        if (run_dir / f"seed_{seed}" / "full" / "metrics.json").is_file():
            print(f"seed {seed} complete; skip", flush=True)
            continue
        cmd = [
            sys.executable,
            "-u",
            str(RUNNER),
            "--subset", "FD002",
            "--seeds", str(seed),
            "--variants", "full", "wo_agcn", "wo_ssm", "wo_physics",
            "--run-dir", str(run_dir),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--batch-size", "256",
            "--window-size", str(locked["window_size"]),
            "--dropout", str(locked["dropout"]),
            "--weight-decay", str(locked["weight_decay"]),
            "--asymmetric-weight", str(full_loss["asym_weight"]),
            "--lambda-mono", str(full_loss["lambda_mono"]),
            "--late-weight", str(full_loss["late_weight"]),
            "--identity-prior-alpha", "0.1",
            "--lstm-hidden-size", "70",
            "--condition-aware",
            "--n-conditions", "6",
            "--clip-test-rul",
            "--num-workers", str(args.num_workers),
            "--device", args.device,
        ]
        log = (log_dir / f"seed_{seed}.log").open("a", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        pending.append((seed, proc, log))
        print(f"launched FD002 seed {seed}", flush=True)
        while sum(1 for _, p, _ in pending if p.poll() is None) >= args.max_parallel:
            time.sleep(10)
    failures = []
    for seed, proc, log in pending:
        code = proc.wait()
        log.close()
        print(f"finished FD002 seed {seed} exit={code}", flush=True)
        if code != 0:
            failures.append(seed)
    if failures:
        print("FAILED seeds:", failures, flush=True)
        return 1
    for path in test_files:
        os.chmod(path, 0o000)
    print("FD002 runs complete and test resealed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
