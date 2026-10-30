#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
mkdir -p logs/order01
run_one() {
  $PY -u scripts/run_candidate.py --subset $1 --variant $2 --seed $3 --split-seed 7 --epochs 18 --patience 8       --d-model 64 --num-layers 1 --out experiments/order01 > logs/order01/$1_$2_$3.log 2>&1
}
export -f run_one
export PY PYTORCH_CUDA_ALLOC_CONF
{
  for ds in FD001 FD002; do
    for v in tf tfA A Aid Arec; do
      for s in 2027 2028 2029; do echo "$ds $v $s"; done
    done
  done
} | xargs -P 4 -n 3 bash -c 'run_one "$@"' _
echo ORDER01_GRID_DONE
