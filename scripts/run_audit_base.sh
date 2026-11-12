#!/usr/bin/env bash
# Launch missing vanilla cells and run the DSPT-Base runtime audit.
set -u
source /media/sangfor/vdb/user/zxk/env.sh
cd "$HC_ROOT"
mkdir -p logs

# 1) missing vanilla cells (FD002-FD004 x 3 seeds x 2 variants = 18)
nohup "$PY" -m paper_b.experiments.run_grid \
  --grid paper_b/experiments/grids/vanilla_missing.json \
  --device cuda:0 --data-dir data/CMAPSS \
  > logs/vanilla_missing.log 2>&1 &
echo "grid_pid=$!"

# 2) runtime audit (CPU, small)
"$PY" -m paper_b.experiments.audit_dspt_base --out /tmp/audit_dspt_base.json \
  > logs/audit_dspt_base.log 2>&1
echo "audit_exit=$?"
tail -5 logs/audit_dspt_base.log
