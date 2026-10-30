#!/bin/bash
exec 8>/tmp/dsh_s9m1.lock
flock -n 8 || exit 0
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for s in 2027 2028 2029; do
  $PY -u scripts/run_candidate.py --subset FD002 --variant A --seed $s --split-seed 9 \
      --ref-mode global --late-weight 1.8 --epochs 30 --patience 10 --d-model 64 --num-layers 1 \
      --out experiments/split9/M1_global > logs/order03/split9_M1_$s.log 2>&1 &
done
wait
echo "S9M1_DONE"
