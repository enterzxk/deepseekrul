#!/bin/bash
# Branch B: oracle shows headroom -> upgrade scale head to attention pooling
# and re-run the full diagnostic set so every arm is compared under the same
# (better) Lambda-hat.  6 arms x 3 folds.
set -u
cd "$HOME/rulb"
mkdir -p experiments/lsc/runs logs/lsc
run_one() {
  local arm=$1 fold=$2
  local out="experiments/lsc/runs/FD002_${arm}_attn_f${fold}_s2027.json"
  if [ -f "$out" ]; then return; fi
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 PYTHONPATH=src \
    /home/user/miniconda3/envs/pmv/bin/python lsc/run.py \
    --subset FD002 --arm "$arm" --fold "$fold" --seed 2027 --epochs 300 \
    --scale_head attn \
    > "logs/lsc/${arm}_attn_f${fold}_s2027.log" 2>&1
}
export -f run_one
for arm in plain rel full both oracle rand; do for f in 0 1 2; do
  echo "$arm $f"
done; done | xargs -P "${1:-4}" -n 2 bash -c 'run_one "$0" "$1"'
echo DONE > logs/lsc/attn.status
