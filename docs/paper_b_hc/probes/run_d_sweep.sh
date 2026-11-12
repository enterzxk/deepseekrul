#!/bin/bash
# Stage D acceptance sweep v2, 2026-11-13. Runs on a40 only.
# v2 adds: normalised k basis + zero-init rate output layer, so B3 no longer
# starts with a 1/65000-scale future rate (which pinned T at K).
set -u
cd /media/sangfor/vdb/user/zxk/deepseek-rul || exit 1
export PYTHONPATH=.
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH:-}
P=/home/yndx/miniconda3/envs/a40_env/bin/python
PROBE=docs/paper_b_hc/probes/d_acceptance.py
LOG=experiments/hc/tmp/d_acceptance_sweep2.log
: > "$LOG"

run() {
  echo "=== $* ===" | tee -a "$LOG"
  $P "$PROBE" "$@" 2>&1 | grep -v '^ *warn' | tee -a "$LOG"
  echo "--- exit ${PIPESTATUS[0]}" | tee -a "$LOG"
}

# 1. corrected init at K=600 (above the longest training lifetime, 543)
run --subsets FD001,FD002,FD003,FD004 --K 600 --tag b0fix2_K600
# 2. B2 fixed rate, same K: B2 and B3 must coincide at init
run --subsets FD001,FD002,FD003,FD004 --K 600 --fixed-rate --tag b0fix2_K600
# 3. the same corrected init at the spec's K=200: does the trap persist?
run --subsets FD001,FD002,FD003,FD004 --K 200 --tag b0fix2_K200
# 4. pre-fix bias (default 0) at K=600: the scale error is independent of K
run --subsets FD001,FD002,FD003,FD004 --K 600 --no-b0 --tag nob0v2_K600
echo "DONE" | tee -a "$LOG"
