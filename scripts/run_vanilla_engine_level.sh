#!/usr/bin/env bash
set -u
source /media/sangfor/vdb/user/zxk/env.sh
cd "$HC_ROOT"
"$PY" -m paper_b.experiments.vanilla_engine_level \
  --root paper_b/results/raw \
  --out paper_b/results/vanilla_engine_level.csv > logs/vanilla_engine_level.log 2>&1
echo "exit=$?"
tail -3 logs/vanilla_engine_level.log
