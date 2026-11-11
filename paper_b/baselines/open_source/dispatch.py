"""Bounded background batches, with one sequential chain per upstream model."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["author", "unified"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    root = Path(a.out_root).resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    invocation = {**vars(a), "started_unix": time.time(), "pid": os.getpid()}
    write_json(root / f"dispatch_{a.protocol}_{os.getpid()}.json", invocation)

    def chain(model):
        done = []
        for subset in a.subsets:
            for seed in a.seeds:
                folder = root / a.protocol / model / subset / f"s{seed}"
                if (folder / "metrics.json").exists():
                    done.append({"model": model, "subset": subset, "seed": seed, "state": "existing_complete"})
                    continue
                log = root / "logs" / f"{a.protocol}_{model}_{subset}_{seed}.log"
                cmd = [sys.executable, "-u", "-m", f"paper_b.baselines.open_source.{a.protocol}",
                       "--model", model, "--subset", subset, "--seed", str(seed),
                       "--data-dir", a.data_dir, "--out-root", str(root)]
                env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "MPLBACKEND": "Agg"}
                with log.open("w", encoding="utf-8") as stream:
                    print("START", " ".join(cmd), flush=True)
                    code = subprocess.call(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
                row = {"model": model, "subset": subset, "seed": seed, "returncode": code, "log": str(log)}
                done.append(row)
                print("FINISH", json.dumps(row), flush=True)
                if code:
                    # Fail once per model/cause, not 12 times with an identical bug.
                    break
            if done[-1].get("returncode", 0):
                break
        write_json(root / f"dispatch_{a.protocol}_{model}.json", done)
        return done

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(chain, a.models))
    write_json(root / f"dispatch_{a.protocol}_finished_{os.getpid()}.json", rows)
    if any(row.get("returncode", 0) for group in rows for row in group):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
