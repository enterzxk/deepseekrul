#!/usr/bin/env bash
# DSPT-Base without cycle encoding (NC_base_L70), same machine as the Vanilla arms.
set -u
source /media/sangfor/vdb/user/zxk/env.sh
cd "$HC_ROOT"
mkdir -p logs
nohup "$PY" -m paper_b.experiments.run_grid \
  --grid paper_b/experiments/grids/nocycle.json \
  --device cuda:0 --data-dir data/CMAPSS \
  --out-root paper_b/results/raw_a40 \
  > logs/audit_nocycle.log 2>&1 &
echo "nocycle_pid=$!"
sleep 2
tail -1 logs/audit_nocycle.log
