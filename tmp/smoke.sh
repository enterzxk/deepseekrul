#!/bin/bash
cd /home/user/rulb
PY=/home/user/miniconda3/envs/pmv/bin/python
for m in dual short long repeat; do
  CUDA_VISIBLE_DEVICES=0 $PY scripts/run_dspt.py FD002 0 --epochs 2 --mode $m --tag SMOKE2 2>&1 | tail -1
done
