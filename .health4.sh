#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
rm -rf experiments/health4
$PY -u scripts/run_candidate.py --subset FD002 --variant A --seed 2027 --split-seed 7 \
  --ref-mode global --epochs 1 --patience 1 --d-model 64 --num-layers 1 \
  --out experiments/health4 > /tmp/h4.log 2>&1
echo exit=$?
tail -n 1 /tmp/h4.log
$PY - <<'PY'
import json, glob
f = glob.glob('experiments/health4/FD002/A_s2027/metrics.json')
if f:
    d = json.load(open(f[0]))
    print('run_hyperparameters:', d.get('run_hyperparameters'))
    print('provenance keys:', sorted(d.get('provenance', {})))
else:
    print('no metrics')
PY
