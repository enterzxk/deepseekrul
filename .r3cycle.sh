#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
mkdir -p experiments/r3_cycle results/dsh/review_r3
for var in A BE; do
  for seed in 2027 2028 2029; do
    out=experiments/r3_cycle/${var}_s${seed}
    if [ -f $out/metrics.json ]; then echo "skip $out"; continue; fi
    $PY -u scripts/run_candidate.py --subset FD002 --variant $var --seed $seed --split-seed 7 \
      --ref-mode global --epochs 30 --patience 30 --d-model 64 --num-layers 1 \
      --out $out > /tmp/r3c_${var}_${seed}.log 2>&1
    echo "$var s$seed exit=$? $(tail -n1 /tmp/r3c_${var}_${seed}.log)"
  done
done
