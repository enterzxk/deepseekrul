"""One launch owns the bounded experiment, gates, and final benchmark report.

No external agent needs to poll job-output. JSON status is updated only when
a stage changes. All child process groups belong to this pipeline; timeouts
terminate only these newly launched jobs, never existing research processes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--cache", default="cache")
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    p.add_argument("--wall-hours", type=float, default=24.)
    p.add_argument("--deadline", help="Timezone-aware ISO timestamp; the earlier limit wins")
    p.add_argument("--prior", choices=("full", "none"), default="full")
    p.add_argument("--skip-official", action="store_true")
    args = p.parse_args()
    if args.wall_hours <= 0 or len(set(args.gpus)) != len(args.gpus):
        p.error("Positive wall-hours and unique GPU identifiers are required")
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    finish = time.time()+args.wall_hours*3600
    if args.deadline:
        d = datetime.fromisoformat(args.deadline)
        if d.tzinfo is None:
            p.error("deadline needs an explicit timezone")
        finish = min(finish, d.timestamp())
    state_path = root/"PIPELINE_STATE.json"
    state = {"status": "STARTING", "pid": os.getpid(), "owned_process_groups": [],
             "deadline_utc": datetime.fromtimestamp(finish, timezone.utc).isoformat(),
             "history": [], "training_performance_claim": False}
    children = []

    def status(name, **extra):
        state.update(status=name, **extra)
        state["history"].append({"status": name, "time": datetime.now(timezone.utc).isoformat()})
        tmp = state_path.with_suffix(".partial")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(state_path)
        print(name, flush=True)

    def terminate_owned():
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        children.clear()
        state["owned_process_groups"] = []

    def execute(commands, label):
        remaining = finish-time.time()
        if remaining <= 0:
            raise TimeoutError("Overall research deadline reached")
        handles = []
        status(label)
        try:
            for i, (command, env) in enumerate(commands):
                handle = (root/f"pipeline_{label}_{i}.log").open("a", encoding="utf-8")
                handles.append(handle)
                child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                          env=env, start_new_session=True)
                children.append(child)
            state["owned_process_groups"] = [c.pid for c in children]
            status(label+"_RUNNING")
            for child in children:
                rc = child.wait(timeout=max(.01, finish-time.time()))
                if rc:
                    raise RuntimeError(f"{label} worker failed with exit {rc}; inspect its saved log")
            children.clear()
            state["owned_process_groups"] = []
        finally:
            for handle in handles:
                handle.close()

    def phase(name):
        commands = []
        for shard, gpu in enumerate(args.gpus):
            command = [sys.executable, "-m", "tim_contrast.queue", "--phase", name,
                       "--data-root", str(Path(args.data_root).resolve()), "--out", str(root),
                       "--cache", str(Path(args.cache).resolve()), "--prior", args.prior,
                       "--gpu", str(gpu), "--shard", str(shard), "--shards", str(len(args.gpus)),
                       "--wall-hours", str(max(.001, (finish-time.time())/3600))]
            commands.append((command, dict(os.environ)))
        execute(commands, name)

    def analyze(name):
        execute([([sys.executable, "-m", "tim_contrast.analyze", "--stage", name,
                  "--out", str(root), "--prior", args.prior], dict(os.environ))], "analyze_"+name)
        file_name = {"screen": "screen", "tail": "screen_risk", "confirm": "confirmation", "support": "support"}[name]
        path = root/"analysis"/(file_name+".json")
        if not path.is_file():
            raise RuntimeError(f"Analysis incomplete: {root/'analysis'}")
        return json.loads(path.read_text(encoding="utf-8"))

    import fcntl
    with (root/"PIPELINE.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            phase("screen")
            result = analyze("screen")
            if result["next"] == "TAIL_SCREEN":
                phase("tail")
                result = analyze("tail")
            if result["next"] != "CONFIRM":
                status("STOP_NO_GAIN", decision=result)
                return
            phase("confirm")
            result = analyze("confirm")
            if result["status"] != "DEVELOPMENT_GATE_PASS":
                status("STOP_NO_GAIN", decision=result)
                return
            phase("ablation")
            phase("guard")
            result = analyze("support")
            if result["status"] != "SUPPORT_PASS":
                status("STOP_MECHANISM_OR_SCOPE", decision=result)
                return
            if args.skip_official:
                status("DEVELOPMENT_COMPLETE_OFFICIAL_NOT_RUN", decision=result)
                return
            phase("benchmark")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpus[0]), OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            for stage in ("predict", "score"):
                if stage == "predict" and (root/"official"/"sealed.json").is_file():
                    continue
                execute([([sys.executable, "-m", "tim_contrast.official", "--stage", stage,
                          "--data-root", str(Path(args.data_root).resolve()), "--out", str(root)], env)],
                        "official_"+stage)
            status("BENCHMARK_EVALUATED_REVIEW_REQUIRED", decision=result)
        except (TimeoutError, subprocess.TimeoutExpired) as exc:
            terminate_owned()
            status("BUDGET_STOP_PARTIAL", reason=str(exc))
        except BaseException as exc:
            terminate_owned()
            status("FAILED_STOP", reason=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
