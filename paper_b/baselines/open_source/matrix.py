"""Finite four-stage experiment matrix; resumes completed/active run IDs.

This is a batch job for the current request, not an automation or service.
Existing training children are adopted read-only until their terminal record.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from .common import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--workers", default=4, type=int)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    lock = threading.Lock()
    blocked = set()
    rows = []

    def task(item):
        protocol, model, subset, seed = item
        version = "v2" if protocol == "author" and model == "aidgn" else ("v1" if protocol == "author" and model in ["two_phase", "dual_mixer"] else "v3")
        out_root = root / version
        folder = out_root / protocol / model / subset / f"s{seed}"
        logdir = out_root / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        with lock:
            if (protocol, model) in blocked:
                return {"task": item, "state": "blocked_by_prior_error"}
        if (folder / "metrics.json").exists():
            return {"task": item, "state": "existing_complete", "path": str(folder)}
        while (folder / "RUNNING.json").exists():
            record = json.loads((folder / "RUNNING.json").read_text())
            try:
                os.kill(record["pid"], 0)
            except ProcessLookupError:
                return {"task": item, "state": "stale_claim", "path": str(folder)}
            time.sleep(5)
            if (folder / "metrics.json").exists():
                return {"task": item, "state": "adopted_complete", "path": str(folder)}
        if (folder / "FAILED.json").exists():
            with lock:
                blocked.add((protocol, model))
            return {"task": item, "state": "existing_failed", "path": str(folder)}
        log = logdir / f"{protocol}_{model}_{subset}_{seed}.log"
        cmd = [sys.executable, "-u", "-m", f"paper_b.baselines.open_source.{protocol}", "--model", model,
               "--subset", subset, "--seed", str(seed), "--data-dir", a.data_dir, "--out-root", str(out_root)]
        print("START", " ".join(cmd), flush=True)
        with log.open("x", encoding="utf-8") as stream:
            code = subprocess.call(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "MPLBACKEND": "Agg"})
        row = {"task": item, "returncode": code, "path": str(folder)}
        if code:
            with lock:
                blocked.add((protocol, model))
        print("FINISH", json.dumps(row), flush=True)
        return row

    stages = [("author", ["aidgn", "two_phase", "dual_mixer"]),
              ("unified", ["aidgn", "two_phase", "dual_mixer"]),
              ("author", ["ffve", "fcstgnn"]), ("unified", ["ffve", "fcstgnn"])]
    for number, (protocol, models) in enumerate(stages, 1):
        print("STAGE", number, protocol, models, flush=True)
        # Stage all registered seed-101 FDs before later seeds; methods
        # interleave so short jobs naturally fill slots around long training.
        tasks = [(protocol, model, subset, seed) for seed in [101, 202, 303]
                 for subset in ["FD001", "FD002", "FD003", "FD004"] for model in models]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            result = list(pool.map(task, tasks))
        rows.extend(result)
        write_json(root / f"matrix_stage_{number}.json", result)
        # Never launch a unified result for an author model whose runnable
        # method has failed; the next independent models still proceed.
        for (p, model) in list(blocked):
            if p == "author":
                blocked.add(("unified", model))
    write_json(root / "matrix_complete.json", rows)
    print("MATRIX_COMPLETE", len(rows), "blocked", sorted(blocked), flush=True)


if __name__ == "__main__":
    main()
