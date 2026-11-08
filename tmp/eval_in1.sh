#!/bin/bash
cd /home/user/rulb
PY=/home/user/miniconda3/envs/pmv/bin/python
for sub in FD001; do
  for fold in 0 1 2 3 4; do
    tag=IN_${sub}_s101_f$fold
    [ -f experiments/$tag/$sub/dspt_s101.csv ] || continue
    ck=$(ls experiments/$tag/$sub/*.pt | head -1)
    $PY scripts/eval_dspt.py --subset $sub --fold $fold --seed 101 --instnorm \
        --ckpt $ck --out experiments/TESTCACHE_IN/${sub}_s101_f$fold.csv 2>&1 | tail -1
  done
done