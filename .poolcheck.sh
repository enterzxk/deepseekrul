#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
rm -rf experiments/poolcheck
$PY -u scripts/run_candidate.py --subsets FD002,FD004 --variant A --seed 2027 --split-seed 7 \
  --ref-mode global --epochs 1 --patience 1 --d-model 64 --num-layers 1 \
  --out experiments/poolcheck > /tmp/pool.log 2>&1
echo exit=$?
tail -n 3 /tmp/pool.log
$PY - <<'EOF'
import json, glob
f = glob.glob('experiments/poolcheck/*/A_s2027/metrics.json')
if f:
    m = json.load(open(f[0]))
    print('subset:', m.get('subset'), '| val_units:', len(m.get('val_units', [])), '| train_units:', m.get('train_units_count'))
EOF
