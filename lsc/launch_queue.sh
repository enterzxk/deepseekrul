#!/bin/bash
# Two-stage decisive queue:
#   stage 1: full+detach x3 folds  -- does an HONEST Lambda-hat (no encoder
#            gradient) fix the co-opting failure?  The decisive test.
#   stage 2: attention-pooling scale head, 6 arms x3 folds -- head upgrade.
set -u
cd "$HOME/rulb"
mkdir -p experiments/lsc/runs logs/lsc
run_one() {
  local arm=$1 fold=$2 extra=$3 tag=$4
  local out="experiments/lsc/runs/FD002_${arm}${tag}_f${fold}_s2027.json"
  if [ -f "$out" ]; then return; fi
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 PYTHONPATH=src \
    /home/user/miniconda3/envs/pmv/bin/python lsc/run.py \
    --subset FD002 --arm "$arm" --fold "$fold" --seed 2027 --epochs 300 $extra \
    > "logs/lsc/${arm}${tag}_f${fold}_s2027.log" 2>&1
}
export -f run_one
# stage 1: detach
for f in 0 1 2; do echo "full $f --detach_lam _det"; done | \
  xargs -P 3 -n 3 bash -c 'run_one "$0" "$1" "$2" "$3"'
echo STAGE1_DONE >> logs/lsc/queue.status
# stage 2: attn head, all six arms
for arm in plain rel full both oracle rand; do for f in 0 1 2; do
  echo "$arm $f --scale_head attn _attn"
done; done | xargs -P 4 -n 4 bash -c 'run_one "$0" "$1" "$2" "$3"'
echo DONE > logs/lsc/queue.status
