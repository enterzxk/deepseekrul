"""Run or reuse the locked five-seed FD001 confirmatory ablation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_ablation.py"
SUMMARIZER = ROOT / "scripts" / "summarize_ablations.py"
LOCK = ROOT / "results" / "locked_hparams.json"
FINAL = ROOT / "results" / "final_FD001_5seeds"
FROZEN = FINAL / "frozen_config.json"
SEEDS = (101, 202, 303, 404, 505)
VARIANTS = ("wo_agcn", "identity_graph", "wo_ssm", "wo_physics", "wo_mono", "full")


def _config_matches(payload: dict) -> bool:
    if not FROZEN.is_file():
        return False
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    return (
        abs(float(frozen["window_size"]) - float(payload["window_size"])) < 1e-9
        and abs(float(frozen["dropout"]) - float(payload["dropout"])) < 1e-9
        and abs(float(frozen["weight_decay"]) - float(payload["weight_decay"])) < 1e-12
        and abs(float(frozen["asym_weight"]) - float(payload["full_loss"]["asym_weight"])) < 1e-9
        and abs(float(frozen["lambda_mono"]) - float(payload["full_loss"]["lambda_mono"])) < 1e-9
        and abs(float(frozen["late_weight"]) - float(payload["full_loss"]["late_weight"])) < 1e-9
    )


def main() -> int:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    payload = {
        "window_size": lock["window_size"],
        "dropout": lock["dropout"],
        "weight_decay": lock["weight_decay"],
        "asym_weight": lock["full_loss"]["asym_weight"],
        "lambda_mono": lock["full_loss"]["lambda_mono"],
        "late_weight": lock["full_loss"]["late_weight"],
    }
    if _config_matches(payload) and (FINAL / "summary.json").is_file():
        print("Existing 5-seed FD001 run matches locked hparams; reuse.", flush=True)
        return 0

    test_files = [ROOT / "data" / "CMAPSS" / f"test_FD001.txt", ROOT / "data" / "CMAPSS" / "RUL_FD001.txt"]
    for path in test_files:
        os.chmod(path, 0o644)
    run_root = ROOT / "experiments" / "final" / "FD001_locked"
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs" / "final_fd001"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for seed in SEEDS:
        run_dir = run_root / f"seed_{seed}"
        if (run_dir / f"seed_{seed}" / "full" / "metrics.json").is_file():
            print(f"seed {seed} complete; skip", flush=True)
            continue
        cmd = [
            sys.executable,
            "-u",
            str(RUNNER),
            "--subset", "FD001",
            "--seeds", str(seed),
            "--variants", *VARIANTS,
            "--run-dir", str(run_dir),
            "--epochs", "80",
            "--patience", "20",
            "--batch-size", "256",
            "--window-size", str(payload["window_size"]),
            "--dropout", str(payload["dropout"]),
            "--weight-decay", str(payload["weight_decay"]),
            "--asymmetric-weight", str(payload["asym_weight"]),
            "--lambda-mono", str(payload["lambda_mono"]),
            "--late-weight", str(payload["late_weight"]),
            "--identity-prior-alpha", "0.1",
            "--lstm-hidden-size", "70",
            "--clip-test-rul",
            "--num-workers", "2",
            "--device", "cuda",
        ]
        log = (log_dir / f"seed_{seed}.log").open("a", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        pending.append((seed, proc, log))
        print(f"launched FD001 seed {seed}", flush=True)
        while sum(1 for _, p, _ in pending if p.poll() is None) >= 5:
            time.sleep(10)
    failures = []
    for seed, proc, log in pending:
        code = proc.wait()
        log.close()
        print(f"finished FD001 seed {seed} exit={code}", flush=True)
        if code != 0:
            failures.append(seed)
    if failures:
        return 1

    FINAL.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(SUMMARIZER),
            *[str(run_root / f"seed_{s}") for s in SEEDS],
            "--output-dir", str(FINAL),
        ],
        cwd=ROOT,
        check=True,
    )
    FROZEN.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for path in test_files:
        os.chmod(path, 0o000)
    print("FD001 locked confirmatory run complete and test resealed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
