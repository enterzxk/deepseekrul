#!/bin/bash
exec 8>/tmp/dsh_shuf2.lock
flock -n 8 || exit 0
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
pids=""
for s in 2027 2028 2029; do
  f="experiments/shuf/A/FD002/A_s$s/metrics.json"
  [ -f "$f" ] && continue
  $PY -u scripts/run_candidate.py --subset FD002 --variant A --seed $s --split-seed 7 \
      --ref-mode global --late-weight 1.8 --shuffle-features --epochs 30 --patience 10 \
      --d-model 64 --num-layers 1 --out experiments/shuf > logs/order03/shuf2_$s.log 2>&1 &
  pids="$pids $!"
done
wait $pids
echo "SHUF2_DONE $(date)"
