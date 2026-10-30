#!/bin/bash
# Unattended ORDER_02 pipeline with resume. Every task is skipped when a
# complete metrics.json already exists.
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
OUT=experiments/order02b
mkdir -p logs/order02b "$OUT"
run_one() {
  ds=$1; v=$2; s=$3
  f="$OUT/$ds/${v}_s${s}/metrics.json"
  if [ -f "$f" ]; then echo "skip $ds $v $s"; return 0; fi
  $PY -u scripts/run_candidate.py --subset $ds --variant $v --seed $s --split-seed 7 \
      --epochs 30 --patience 10 --d-model 64 --num-layers 1 --out $OUT \
      > logs/order02b/${ds}_${v}_${s}.log 2>&1
}
export -f run_one
export PY PYTORCH_CUDA_ALLOC_CONF OUT
{
  for v in tf tfA A Aid S2 S3 S4 RACDR; do for s in 2027 2028; do echo "FD002 $v $s"; done; done
  for v in tf tfA A Aid S2 RACDR; do for s in 2027 2028; do echo "FD001 $v $s"; done; done
} | xargs -P 3 -n 3 bash -c 'run_one "$@"' _
echo ORDER02B_DONE
