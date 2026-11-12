#!/usr/bin/env bash
# Launch the two missing feature-decomposition vanilla arms (24 cells).
set -u
source /media/sangfor/vdb/user/zxk/env.sh
cd "$HC_ROOT"
mkdir -p logs
nohup "$PY" -m paper_b.experiments.run_grid \
  --grid paper_b/experiments/grids/vanilla_features2.json \
  --device cuda:0 --data-dir data/CMAPSS \
  > logs/vanilla_features2.log 2>&1 &
echo "grid2_pid=$!"
sleep 3
tail -3 logs/vanilla_features2.log
