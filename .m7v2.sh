#!/bin/bash
exec 8>/tmp/dsh_one.lock
flock -n 8 || exit 0
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for s in 2027 2028 2029; do
  f="experiments/order03/M7_rawback/FD002/base_s$s/metrics.json"
  if [ ! -f "$f" ]; then
    $PY -u scripts/run_candidate.py --subset FD002 --variant base --seed $s --split-seed 7 \
        --ref-mode global --late-weight 1.8 --epochs 30 --patience 10 --d-model 64 --num-layers 1 \
        --out experiments/order03/M7_rawback > logs/order03/M7v2_$s.log 2>&1 &
  fi
done
wait
echo "M7V2_DONE $(date)"
