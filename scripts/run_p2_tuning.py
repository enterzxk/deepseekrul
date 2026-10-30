"""Launch the P2 validation-only loss grid with resume."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUNER = ROOT / "scripts" / "tune_validation.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked", type=Path, default=ROOT / "results" / "locked_hparams.json")
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    locked = json.loads(args.locked.read_text(encoding="utf-8"))
    log_dir = ROOT / "logs" / "loss_tuning"
    log_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("p2_mse", "mse", 1.0, 0.0, 2.0),
        ("p2_asym_l15_w05", "asym", 0.5, 0.0, 1.5),
        ("p2_asym_l20_w05", "asym", 0.5, 0.0, 2.0),
        ("p2_asym_l20_w10", "asym", 1.0, 0.0, 2.0),
        ("p2_mono_l005", "mono", 0.0, 0.05, 2.0),
        ("p2_mono_l010", "mono", 0.0, 0.10, 2.0),
        ("p2_full_l20_w05_m005", "full", 0.5, 0.05, 2.0),
        ("p2_full_l20_w10_m005", "full", 1.0, 0.05, 2.0),
        ("p2_full_l20_w05_m010", "full", 0.5, 0.10, 2.0),
    ]

    pending = []
    for tag, mode, asym, mono, late in configs:
        cmd = [
            sys.executable,
            "-u",
            str(TUNER),
            "--tag", tag,
            "--window-size", str(locked["window_size"]),
            "--dropout", str(locked["dropout"]),
            "--weight-decay", str(locked["weight_decay"]),
            "--loss-mode", mode,
            "--asymmetric-weight", str(asym),
            "--lambda-mono", str(mono),
            "--late-weight", str(late),
            "--seeds", "42", "67", "123",
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
            "--val-only",
            "--resume",
        ]
        log = (log_dir / f"{tag}.log").open("a", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        pending.append((tag, proc, log))
        print(f"launched {tag} pid={proc.pid}", flush=True)
        while sum(1 for _, p, _ in pending if p.poll() is None) >= args.max_parallel:
            time.sleep(5)

    failures = []
    for tag, proc, log in pending:
        code = proc.wait()
        log.close()
        print(f"finished {tag} exit={code}", flush=True)
        if code != 0:
            failures.append(tag)
    if failures:
        print("FAILED:", failures, flush=True)
        return 1
    print("P2 grid complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
