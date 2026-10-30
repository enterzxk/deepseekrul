#!/bin/bash
exec 8>/tmp/dsh_shuf4.lock
flock -n 8 || { echo "ALREADY_RUNNING"; exit 0; }
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
echo "START $(date)" > /home/yndx/p8/shuf4.state
pids=""
for s in 2027 2028 2029; do
  f="experiments/shuf/A/FD002/A_s$s/metrics.json"
  if [ -f "$f" ]; then echo "skip $s" >> /home/yndx/p8/shuf4.state; continue; fi
  $PY -u scripts/run_candidate.py --subset FD002 --variant A --seed $s --split-seed 7 \
      --ref-mode global --late-weight 1.8 --shuffle-features --epochs 30 --patience 10 \
      --d-model 64 --num-layers 1 --out experiments/shuf > logs/order03/shuf4_$s.log 2>&1 &
  pids="$pids $!"
done
wait $pids
echo "DONE $(date)" >> /home/yndx/p8/shuf4.state
