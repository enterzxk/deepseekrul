"""Bounded supervisor for four disjoint development-subset controllers.

Only child PIDs created here can be terminated. Other GPU jobs are untouched.
This is an experiment job, not a scheduled app automation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .prepare import digest, dump_new
from .run import execution_hash, load_cache


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True); p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--git-commit", required=True)
    args = p.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.git_commit) is None:
        raise ValueError("full source commit required")
    protocol_path = Path(__file__).parent/"PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    execution = execution_hash()
    preflight = json.loads(args.preflight.read_text()); calibration = json.loads(args.calibration.read_text())
    for evidence in (preflight, calibration):
        if evidence["status"] != "PASS" or evidence["execution_sha256"] != execution or evidence["optimizer_updates"] != 0:
            raise AssertionError("stale or failed launch evidence")
    if not calibration["admission"]:
        raise AssertionError("hardware budget was not admitted")
    for subset in protocol["subsets"]:
        load_cache(args.cache, subset)  # verify every cache and data boundary before starting any fit
    args.output.mkdir(parents=True, exist_ok=False)
    runs = args.output/"runs"; runs.mkdir()
    manifest = {"started_utc": now(), "pid": os.getpid(), "source_git_commit": args.git_commit,
                "execution_sha256": execution, "protocol_sha256": digest(protocol_path),
                "preflight_sha256": digest(args.preflight), "calibration_sha256": digest(args.calibration),
                "expected_training_runs": protocol["training_runs"], "subsets": protocol["subsets"],
                "strategy": "four disjoint subset workers sharing one GPU; fixed identical training protocol",
                "maximum_wall_hours_on_one_shared_gpu": protocol["maximum_gpu_training_hours"],
                "official_test_read": False, "python": sys.version,
                "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    dump_new(args.output/"LAUNCH_MANIFEST.json", manifest)
    processes = {}; handles = []
    start = time.monotonic()
    failure = None
    try:
        for subset in protocol["subsets"]:
            log = open(args.output/f"worker_{subset}.log", "x", encoding="utf-8"); handles.append(log)
            command = [sys.executable, "-B", "-m", "paper_b.ncmapss_response.run", "--cache", str(args.cache),
                       "--output", str(runs), "--subsets", subset]
            processes[subset] = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        dump_new(args.output/"WORKER_PIDS.json", {k: v.pid for k, v in processes.items()})
        while True:
            states = {k: v.poll() for k, v in processes.items()}
            status = {"updated_utc": now(), "workers": states,
                      "sealed_training_runs": sum(1 for _ in runs.glob("*/SUCCESS.json")),
                      "expected_training_runs": protocol["training_runs"],
                      "elapsed_seconds": time.monotonic()-start, "execution_sha256": execution}
            temporary = args.output/".status.next"
            temporary.write_text(json.dumps(status, indent=2)+"\n", encoding="utf-8")
            os.replace(temporary, args.output/"STATUS.json")
            if any(code is not None and code != 0 for code in states.values()):
                raise RuntimeError(f"subset worker failed: {states}")
            if all(code == 0 for code in states.values()):
                break
            if time.monotonic()-start > protocol["maximum_gpu_training_hours"]*3600:
                raise TimeoutError("whole job reached its shared-GPU wall-time budget")
            time.sleep(30)
        actual = {p.parent.name for p in runs.glob("*/SUCCESS.json")}
        expected = set()
        for subset in protocol["subsets"]:
            data, _ = load_cache(args.cache, subset)
            for mode in protocol["modes"]:
                for seed in protocol["seeds"]:
                    for unit in sorted(set(map(int, data["unit"]))):
                        expected.add(f"{subset}_{mode}_u{unit}_s{seed}")
        if actual != expected or len(actual) != protocol["training_runs"]:
            raise AssertionError("completed grid differs from pre-registration")
        dump_new(args.output/"ALL_TRAINING_COMPLETE.json", {"execution_sha256": execution,
                 "training_runs": len(actual), "finished_utc": now(), "elapsed_seconds": time.monotonic()-start})
        with open(args.output/"summary.log", "x", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-B", "-m", "paper_b.ncmapss_response.summarize", "--cache", str(args.cache),
                                     "--runs", str(runs), "--output", str(args.output/"summary")], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError("evidence collector failed; training artifacts are preserved")
        dump_new(args.output/"EXPERIMENT_COLLECTION_COMPLETE.json", {"execution_sha256": execution, "finished_utc": now(),
                  "method_success": "read pre-registered gates in summary; collection success alone proves no benefit"})
    except BaseException as e:
        failure = e
        dump_new(args.output/f"SUPERVISOR_FAILURE_{int(time.time())}.json", {"type": type(e).__name__, "message": str(e), "when": now()})
    finally:
        if failure is not None:
            for process in processes.values():
                if process.poll() is None:
                    process.terminate()
            for process in processes.values():
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
        for handle in handles:
            handle.close()
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
