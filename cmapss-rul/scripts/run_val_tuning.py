"""Launch the P1 validation-only grid with resume and bounded parallelism."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUNER = ROOT / "scripts" / "tune_validation.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("p1",), default="p1")
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 67, 123])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.phase != "p1":
        raise ValueError("This launcher currently implements P1 only.")
    log_dir = ROOT / "logs" / "val_tuning"
    log_dir.mkdir(parents=True, exist_ok=True)

    grid = []
    for window, dropout, wd in itertools.product(
        (30, 50), (0.15, 0.25), ("1e-4", "1e-3")
    ):
        tag = f"p1_w{window}_d{str(dropout).replace('.', '')}_wd{wd}"
        grid.append((tag, window, dropout, wd))
    # The prior locked candidate is included as a control so P3 can reuse it
    # only if it remains the validation winner.
    grid.append(("p1a_w50_d02_wd1e4", 50, 0.2, "1e-4"))

    pending = []
    for tag, window, dropout, wd in grid:
        cmd = [
            sys.executable,
            "-u",
            str(TUNER),
            "--tag", tag,
            "--window-size", str(window),
            "--dropout", str(dropout),
            "--weight-decay", str(wd),
            "--loss-mode", "asym",
            "--asymmetric-weight", "1.0",
            "--seeds", *[str(x) for x in args.seeds],
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
            "--val-only",
            "--resume",
        ]
        log_path = log_dir / f"{tag}.log"
        log = log_path.open("a", encoding="utf-8")
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
    print("P1 grid complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
