#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for arm in est est_rate; do
  out=experiments/r3_v3/${arm}_s2029
  mkdir -p $out
  nohup $PY -u scripts/dsh_r3_four_arms_v2.py --arm $arm --seed 2029 --epochs 30 --out $out > /tmp/r3v3b_${arm}_2029.log 2>&1 &
done
wait
echo "batch 2029 done" >> /tmp/r3v3_status.txt
