#!/bin/bash
exec 8>/tmp/dsh_r2.lock
flock -n 8 || { echo ALREADY; exit 0; }
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
echo "START $(date)" > /home/yndx/p8/r2.state
for s in 2027 2028 2029; do
  $PY -u scripts/run_candidate.py --subset FD002 --variant A --seed $s --split-seed 7 \
      --ref-mode cond --norm-extras --late-weight 1.8 --epochs 30 --patience 10 \
      --d-model 64 --num-layers 1 --out experiments/M3z > logs/order03/M3z_$s.log 2>&1 &
done
wait
echo "DONE $(date)" >> /home/yndx/p8/r2.state
