#!/bin/bash
cd /home/user/rulb
PY=/home/user/miniconda3/envs/pmv/bin/python
for m in 0 0.2; do
  for f in 0 1 2 3 4; do
    ck=experiments/tim_FD002_w100_m${m}_f${f}/FD002/PE_s2027/best_model.pt
    [ -f "$ck" ] || { echo "missing $ck"; continue; }
    $PY scripts/eval_official.py --subset FD002 --fold $f --window 100 --seed 2027 --ckpt "$ck" --out /tmp/p100_m${m}_f${f}.csv 2>&1 | grep RMSE
  done
done
