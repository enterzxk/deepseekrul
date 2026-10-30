#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
mkdir -p logs/order02
run_one() {
  $PY -u scripts/run_candidate.py --subset $1 --variant $2 --seed $3 --split-seed 7 --epochs 24 --patience 10       --d-model 64 --num-layers 1 --out experiments/order02 > logs/order02/$1_$2_$3.log 2>&1
}
export -f run_one
export PY PYTORCH_CUDA_ALLOC_CONF
{
  for ds in FD001 FD002; do
    for v in tf tfA A Aid S2; do
      for s in 2027 2028; do echo "$ds $v $s"; done
    done
  done
} | xargs -P 3 -n 3 bash -c 'run_one "$@"' _
echo ORDER02_GRID_DONE
