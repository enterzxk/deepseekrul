#!/bin/bash
cd /home/user/rulb
PY=/home/user/miniconda3/envs/pmv/bin/python
for spec in "FD003 0 M_FD003_s101_f0" "FD001 0 M_FD001_s101_f0"; do
  set -- $spec
  sub=$1; fold=$2; tag=$3
  ck=$(ls experiments/$tag/$sub/*.pt | head -1)
  for k in 1 3 5 10; do
    $PY scripts/eval_tta.py --subset $sub --fold $fold --ckpt $ck --tta $k --out /tmp/tta_${sub}_${k}.csv
  done
done