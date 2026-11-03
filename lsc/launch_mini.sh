#!/bin/bash
# FD002 mini-matrix driver.  Usage: launch_mini.sh <parallel>
set -u
cd "$HOME/rulb"
mkdir -p experiments/lsc/runs logs/lsc
for v in plain rel full both; do for f in 0 1 2; do
  echo "$v $f"
done; done | xargs -P "${1:-4}" -n 2 sh -c '
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 PYTHONPATH=src /home/user/miniconda3/envs/pmv/bin/python lsc/run.py --subset FD002 --arm "$0" --fold "$1" --seed 2027 --epochs 300 > "logs/lsc/${0}_f${1}_s2027.log" 2>&1'
echo MINI_DONE > logs/lsc/mini.status
