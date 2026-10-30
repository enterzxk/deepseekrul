#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
rm -rf experiments/health
$PY -u scripts/run_candidate.py --subset FD002 --variant A --seed 2027 --split-seed 7 \
  --ref-mode global --epochs 1 --patience 1 --d-model 64 --num-layers 1 \
  --out experiments/health > /tmp/health.log 2>&1
echo exit=$?
tail -n 2 /tmp/health.log
$PY - <<'EOF'
import json, glob
f = glob.glob('experiments/health/FD002/A_s2027/metrics.json')
if f:
    m = json.load(open(f[0]))
    print('subset:', m['subset'], '| variant:', m['variant'], '| params:', m['params'],
          '| provenance:', bool(m.get('provenance')), '| val_units:', len(m.get('val_units', [])))
else:
    print('NO METRICS')
EOF
