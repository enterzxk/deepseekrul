#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
mkdir -p experiments/r3_pe results/dsh/review_r3
for arm in cycle zero bounded; do
  for seed in 2027 2028 2029; do
    out=experiments/r3_pe/${arm}_s${seed}
    if [ -f $out/metrics.json ]; then echo "skip $out"; continue; fi
    $PY scripts/dsh_r3_pe_neural.py --arm $arm --seed $seed --epochs 30 --out $out 2>&1 | tail -2
  done
done
echo '--- summary ---'
$PY - <<'PY'
import json, glob
import numpy as np, pandas as pd
rows = []
for f in sorted(glob.glob('experiments/r3_pe/*/metrics.json')):
    d = json.load(open(f)); rows.append(d)
t = pd.DataFrame(rows)
if len(t):
    print(t.groupby('arm')[['rmse','mae','nasa_mean','late','severe_e10','best_val_rmse']].agg(['mean','std']).round(3).to_string())
    t.to_csv('results/dsh/review_r3/neural_time_arms.csv', index=False)
PY
