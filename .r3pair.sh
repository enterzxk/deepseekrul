#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for var in A BE; do
  out=experiments/r3_cycle/${var}_s2027
  mkdir -p $out
  nohup $PY -u scripts/run_candidate.py --subset FD002 --variant $var --seed 2027 --split-seed 7 \
    --ref-mode global --epochs 30 --patience 30 --d-model 64 --num-layers 1 \
    --out $out > /tmp/r3c_${var}_2027.log 2>&1 &
  echo "launched $var pid=$!"
done
sleep 5
pgrep -c -f run_candidate
