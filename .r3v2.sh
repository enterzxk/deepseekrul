#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
rm -f /tmp/r3v2_status.txt
for arm in est est_rate pooled; do
  for seed in 2027 2028 2029; do
    out=experiments/r3_v2/${arm}_s${seed}
    mkdir -p $out
    nohup $PY -u scripts/dsh_r3_four_arms.py --arm $arm --seed $seed --epochs 30 --out $out > /tmp/r3v2_${arm}_${seed}.log 2>&1 &
  done
  wait
  echo "batch $arm done" >> /tmp/r3v2_status.txt
done
echo ALLDONE >> /tmp/r3v2_status.txt
