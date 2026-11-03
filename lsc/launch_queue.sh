#!/bin/bash
# Two-stage decisive queue (fixed: single |-packed argument per job).
#   stage 1: full+detach x3 folds  -- honest Lambda-hat (no encoder gradient)
#   stage 2: attention-pooling scale head, 6 arms x3 folds
set -u
cd "$HOME/rulb"
mkdir -p experiments/lsc/runs logs/lsc
run_one() {
  local spec="$1"
  local arm fold extra tag
  IFS='|' read -r arm fold extra tag <<< "$spec"
  local out="experiments/lsc/runs/FD002_${arm}${tag}_f${fold}_s2027.json"
  if [ -f "$out" ]; then return; fi
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 PYTHONPATH=src \
    /home/user/miniconda3/envs/pmv/bin/python lsc/run.py \
    --subset FD002 --arm "$arm" --fold "$fold" --seed 2027 --epochs 300 $extra \
    > "logs/lsc/${arm}${tag}_f${fold}_s2027.log" 2>&1
}
export -f run_one
for f in 0 1 2; do echo "full|$f|--detach_lam|_det"; done | \
  xargs -P 3 -n 1 bash -c 'run_one "$0"'
echo STAGE1_DONE >> logs/lsc/queue.status
for arm in plain rel full both oracle rand; do for f in 0 1 2; do
  echo "${arm}|$f|--scale_head attn|_attn"
done; done | xargs -P 4 -n 1 bash -c 'run_one "$0"'
echo DONE > logs/lsc/queue.status
