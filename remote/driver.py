#!/usr/bin/env python3
"""Serial task-queue driver. Task file format: LOGNAME<TAB>CMD (one per line).
Logs are written next to the status file. Env (CUDA_VISIBLE_DEVICES) is
inherited from the driver process, so one driver = one GPU."""
import os
import shlex
import subprocess
import sys
import time


def main():
    task_file, status_file = sys.argv[1], sys.argv[2]
    log_dir = os.path.dirname(os.path.abspath(status_file))
    with open(task_file, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f
                 if l.strip() and not l.strip().startswith("#")]
    with open(status_file, "w", encoding="utf-8") as f:
        f.write(f"RUNNING 0/{len(lines)}\n")
    for i, line in enumerate(lines):
        logname, cmd = line.split("\t", 1)
        logpath = os.path.join(log_dir, logname)
        t0 = time.time()
        with open(logpath, "w", encoding="utf-8") as lf:
            p = subprocess.run(shlex.split(cmd), stdout=lf,
                               stderr=subprocess.STDOUT,
                               cwd=os.path.expanduser("~/rulb"))
        dt = time.time() - t0
        with open(status_file, "a", encoding="utf-8") as f:
            f.write(f"{i + 1}/{len(lines)} rc={p.returncode} "
                    f"{int(dt)}s {logname}\n")
        if p.returncode != 0:
            with open(status_file, "a", encoding="utf-8") as f:
                f.write(f"FAIL {logname}\n")
    with open(status_file, "a", encoding="utf-8") as f:
        f.write("DONE\n")


if __name__ == "__main__":
    main()
