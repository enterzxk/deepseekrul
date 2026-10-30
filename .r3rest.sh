#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
ls experiments/r3_cycle/*/metrics.json 2>/dev/null | wc -l
for seed in 2028 2029; do
  for var in A BE; do
    out=experiments/r3_cycle/${var}_s${seed}
    mkdir -p $out
    nohup $PY -u scripts/run_candidate.py --subset FD002 --variant $var --seed $seed --split-seed 7 \
      --ref-mode global --epochs 30 --patience 30 --d-model 64 --num-layers 1 \
      --out $out > /tmp/r3c_${var}_${seed}.log 2>&1 &
    echo "launched $var s$seed"
  done
done
sleep 4; pgrep -c -f run_candidate
