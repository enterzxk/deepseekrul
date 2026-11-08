#!/bin/bash
cd /home/user/rulb
PY=/home/user/miniconda3/envs/pmv/bin/python
mkdir -p experiments/TESTCACHE_IN
for fold in 0 1 2 3 4; do
  tag=IN_FD003_s101_f$fold
  ck=$(ls experiments/$tag/FD003/*.pt | head -1)
  $PY scripts/eval_dspt.py --subset FD003 --fold $fold --seed 101 --instnorm \
      --ckpt $ck --out experiments/TESTCACHE_IN/FD003_s101_f$fold.csv 2>&1 | tail -1
done