#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
mkdir -p logs/trackA
run_one() {
  $PY -u scripts/run_candidate.py --subset $1 --variant $2 --seed $3 --epochs 15 --patience 8       --d-model 64 --num-layers 1 --out experiments/trackA > logs/trackA/$1_$2_$3.log 2>&1
}
export -f run_one
export PY PYTORCH_CUDA_ALLOC_CONF
{
  for v in base B E C BE BC EC A tf; do for s in 2027 2028; do echo "FD001 $v $s"; done; done
  for v in A tf; do for s in 2027 2028; do echo "FD002 $v $s"; done; done
} | xargs -P 4 -n 3 bash -c 'run_one "$@"' _
echo TRACK_A_ROUND1_DONE
