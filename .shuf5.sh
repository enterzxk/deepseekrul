#!/bin/bash
exec 8>/tmp/dsh_shuf5.lock
flock -n 8 || { echo "ALREADY"; exit 0; }
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
echo "START $(date)" > /home/yndx/p8/shuf5.state
pids=""
for s in 2027 2028 2029; do
  $PY -u scripts/run_candidate.py --subset FD001 --variant A --seed $s --split-seed 7 \
      --ref-mode global --late-weight 1.8 --shuffle-features --epochs 30 --patience 10 \
      --d-model 64 --num-layers 1 --out experiments/shuf_fd001 > logs/order03/shuf5_$s.log 2>&1 &
  pids="$pids $!"
done
wait $pids
echo "DONE $(date)" >> /home/yndx/p8/shuf5.state
