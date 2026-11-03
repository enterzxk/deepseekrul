#!/bin/bash
# Diagnostic arms: oracle (true lifetime, upper bound) and rand (noise control).
# Usage: launch_diag.sh <parallel>
set -u
cd "$HOME/rulb"
mkdir -p experiments/lsc/runs logs/lsc
run_one() {
  local arm=$1 fold=$2
  local out="experiments/lsc/runs/FD002_${arm}_f${fold}_s2027.json"
  if [ -f "$out" ]; then return; fi
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 PYTHONPATH=src \
    /home/user/miniconda3/envs/pmv/bin/python lsc/run.py \
    --subset FD002 --arm "$arm" --fold "$fold" --seed 2027 --epochs 300 \
    > "logs/lsc/${arm}_f${fold}_s2027.log" 2>&1
}
export -f run_one
for arm in oracle rand; do for f in 0 1 2; do
  echo "$arm $f"
done; done | xargs -P "${1:-4}" -n 2 bash -c 'run_one "$0" "$1"'
echo DONE > logs/lsc/diag.status
