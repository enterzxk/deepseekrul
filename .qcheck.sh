#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
rm -rf experiments/qcheck
$PY -u scripts/run_candidate.py --subset FD002 --variant A --seed 2027 --split-seed 7 \
  --ref-mode global --quantile 0.25 --epochs 1 --patience 1 --d-model 64 --num-layers 1 \
  --out experiments/qcheck > /tmp/q.log 2>&1
echo exit=$?
tail -n 3 /tmp/q.log
$PY - <<'EOF'
import json, glob
f = sorted(glob.glob('experiments/qcheck/FD002/*/metrics.json'))
if f:
    m = json.load(open(f[0])); print('val:', round(m['best_val_rmse'],2), '| provenance:', bool(m.get('provenance')))
EOF
