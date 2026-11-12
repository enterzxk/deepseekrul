#!/usr/bin/env bash
# Same-machine re-run of the two original Vanilla FD001 arms (6 cells).
set -u
source /media/sangfor/vdb/user/zxk/env.sh
cd "$HC_ROOT"
mkdir -p logs
nohup "$PY" -m paper_b.experiments.run_grid \
  --grid paper_b/experiments/grids/audit_van_fd001_a40.json \
  --device cuda:0 --data-dir data/CMAPSS \
  --out-root paper_b/results/raw_a40 \
  > logs/audit_van_fd001_a40.log 2>&1 &
echo "pid=$!"
sleep 2
tail -1 logs/audit_van_fd001_a40.log
