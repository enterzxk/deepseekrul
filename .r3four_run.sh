#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
rm -f /tmp/r3four_status.txt
jobs=0
for arm in cycle none oracle est; do
  for seed in 2027 2028 2029; do
    out=experiments/r3_four/${arm}_s${seed}
    mkdir -p $out
    if [ -f $out/metrics.json ]; then echo "skip $arm $seed" >> /tmp/r3four_status.txt; continue; fi
    nohup $PY -u scripts/dsh_r3_four_arms.py --arm $arm --seed $seed --epochs 30 --out $out > /tmp/r3f_${arm}_${seed}.log 2>&1 &
    jobs=$((jobs+1))
    if [ $((jobs % 3)) -eq 0 ]; then wait; fi
  done
done
wait
echo ALLDONE >> /tmp/r3four_status.txt
