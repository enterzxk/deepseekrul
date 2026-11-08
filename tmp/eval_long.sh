#!/bin/bash
cd /home/user/rulb
PY=/home/user/miniconda3/envs/pmv/bin/python
for f in 2 3; do
  $PY scripts/eval_dspt.py --subset FD002 --fold $f --seed 101 --layers 2 \
    --ckpt experiments/D1_FD002_long_f${f}/FD002/dspt_s101.pt --out /tmp/te_long_f${f}.csv 2>&1 | grep RMSE
done
