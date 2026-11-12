#!/usr/bin/env bash
# Same-machine reproducibility set: DSPT-Base x 4 subsets x 3 seeds, plus the two
# original Vanilla FD001 arms, all into paper_b/results/raw_a40 (no overwrite).
set -u
source /media/sangfor/vdb/user/zxk/env.sh
cd "$HC_ROOT"
mkdir -p logs
nohup "$PY" -m paper_b.experiments.run_grid \
  --grid paper_b/experiments/grids/audit_same_machine.json \
  --device cuda:0 --data-dir data/CMAPSS \
  --out-root paper_b/results/raw_a40 \
  > logs/audit_same_machine.log 2>&1 &
echo "same_machine_pid=$!"
sleep 3
tail -2 logs/audit_same_machine.log
