#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for seed in 2027 2028 2029; do
  out=experiments/r3_four/est_s${seed}
  mkdir -p $out
  nohup $PY -u scripts/dsh_r3_four_arms.py --arm est --seed $seed --epochs 30 --out $out > /tmp/r3f_est_${seed}.log 2>&1 &
done
wait
echo ALLESTDONE
