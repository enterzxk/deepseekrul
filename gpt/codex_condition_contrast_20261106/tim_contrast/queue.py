"""Run a bounded queue; print only completion/failure, not polled job output."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def underlying(label):
    risk = label.endswith("_risk")
    suffix = "_risk" if risk else ""
    if label.startswith("scheduled"):
        return ["pe50"+suffix, "w100_late"+suffix]
    return [label]


def plan(phase, out, prior):
    from .run import code_fingerprint
    if phase in ("screen", "tail"):
        risk = phase == "tail"
        if risk:
            gate = json.loads((Path(out)/"analysis"/"screen.json").read_text(encoding="utf-8"))
            if (gate.get("next") != "TAIL_SCREEN" or gate["prior"] != prior
                    or gate["code_sha256"] != code_fingerprint()):
                raise ValueError("Tail rescue is allowed only after the frozen screen calls for it")
        arms = [x+("_risk" if risk else "") for x in
                ("pe50", "pe100", "w100_late", "fixed_trend", "learned_trend")]
        return [(s, a, f, 101) for s in ("FD002", "FD004") for f in (0, 2) for a in arms]
    gate = json.loads((Path(out)/"analysis"/"promotion.json").read_text(encoding="utf-8"))
    if (gate["next"] != "CONFIRM" or gate["prior"] != prior
            or gate["code_sha256"] != code_fingerprint()):
        raise ValueError("No valid promotion")
    if phase == "confirm":
        jobs = []
        for s, refs in gate["references"].items():
            labels = list(dict.fromkeys(y for x in [gate["candidate"]]+refs for y in underlying(x)))
            jobs.extend((s, a, f, seed) for seed in (101, 202, 303) for f in range(5) for a in labels)
        return jobs
    confirmed = json.loads((Path(out)/"analysis"/"confirmation.json").read_text(encoding="utf-8"))
    if confirmed["status"] != "DEVELOPMENT_GATE_PASS":
        raise ValueError("Ablations/guards require the confirmation gate")
    suffix = "_risk" if gate["candidate"].endswith("_risk") else ""
    if phase == "benchmark":
        support = json.loads((Path(out)/"analysis"/"support.json").read_text(encoding="utf-8"))
        if support["status"] != "SUPPORT_PASS" or support["code_sha256"] != code_fingerprint():
            raise ValueError("Complete the support gate before filling official benchmark models")
        return [(s, label, f, seed) for s, check in support["guard_comparisons"].items()
                for seed in (101, 202, 303) for f in range(5)
                for label in (gate["candidate"], check["reference"])]
    if phase == "ablation":
        return [(s, "unmatched_trend"+suffix, f, 101) for s in ("FD002", "FD004") for f in range(5)]
    return [(s, a+suffix, f, 101) for s in ("FD001", "FD003") for f in range(5)
            for a in ("pe50", "pe100", "fixed_trend", "learned_trend")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("screen", "tail", "confirm", "ablation", "guard", "benchmark"), required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--cache", default="cache")
    p.add_argument("--prior", choices=("full", "none"), default="full")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, default=2)
    p.add_argument("--wall-hours", type=float, default=8.)
    args = p.parse_args()
    if args.wall_hours <= 0 or not 0 <= args.shard < args.shards:
        p.error("Positive wall-hours and 0 <= shard < shards are required")
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root/"queue_logs"
    logs.mkdir(exist_ok=True)
    jobs = plan(args.phase, root, args.prior)
    # Entire subsets stay on one shard: no two workers populate the same cache.
    subsets = sorted(set(j[0] for j in jobs))
    jobs = [j for j in jobs if subsets.index(j[0]) % args.shards == args.shard]
    manifest = logs/f"{args.phase}_{args.prior}_shard{args.shard}.json"
    status = {"phase": args.phase, "jobs": jobs, "completed": [], "failed": [], "status": "RUNNING"}
    lock = logs/f"gpu{args.gpu}.lock"
    # flock is held by this worker throughout the queue. It is released by
    # the OS if the worker exits; no detached duplicate worker is submitted.
    import fcntl
    with lock.open("a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        for subset, label, fold, seed in jobs:
            if time.monotonic()-start >= args.wall_hours*3600:
                status["status"] = "BUDGET_STOP_PARTIAL"
                break
            risk = label.endswith("_risk")
            arm = label.removesuffix("_risk")
            cmd = [sys.executable, "-m", "tim_contrast.run", "--data-root", str(Path(args.data_root).resolve()),
                   "--out", str(root), "--cache", str(Path(args.cache).resolve()), "--subset", subset,
                   "--arm", arm, "--fold", str(fold), "--seed", str(seed), "--prior", args.prior,
                   "--nasa-weight", "1.0" if risk else "0.0", "--device", "cuda"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
            job_id = f"{subset}_{label}_f{fold}_s{seed}_{args.prior}"
            manifest.write_text(json.dumps(status, indent=2), encoding="utf-8")
            with (logs/f"{job_id}.log").open("a", encoding="utf-8") as log:
                rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
            if rc:
                status["failed"].append({"job": job_id, "returncode": rc})
                status["status"] = "FAILED_STOP"
                print(f"FAILED {job_id}; see {logs/job_id}.log", flush=True)
                break
            status["completed"].append(job_id)
            print(f"DONE {len(status['completed'])}/{len(jobs)} {job_id}", flush=True)
        else:
            status["status"] = "COMPLETE"
        status["wall_seconds"] = time.monotonic()-start
        manifest.write_text(json.dumps(status, indent=2), encoding="utf-8")
    if status["status"] == "FAILED_STOP":
        raise SystemExit(1)
    print(status["status"], flush=True)


if __name__ == "__main__":
    main()
