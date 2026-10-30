#!/bin/bash
# Persistent single-instance cell runner.
# Usage: supervisor.sh <queue-file>; appending lines to the queue is picked up
# on the next pass. A flock guarantees exactly one supervisor.
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
QUEUE=${1:-/home/yndx/p8/queue.txt}
exec 9>/tmp/dsh_supervisor.lock
flock -n 9 || { echo "supervisor already running"; exit 0; }
mkdir -p logs/order03
while true; do
  pending=0
  while read -r tag ds v ref s; do
    [ -z "$tag" ] && continue
    f="experiments/order03/$tag/$ds/${v}_s${s}/metrics.json"
    [ -f "$f" ] && continue
    pending=1
    if [ "$(pgrep -c -f '[r]un_candidate')" -lt 3 ]; then
      nohup $PY -u scripts/run_candidate.py --subset "$ds" --variant "$v" --seed "$s" \
        --split-seed 7 --ref-mode "$ref" --late-weight 1.8 --epochs 30 --patience 10 \
        --d-model 64 --num-layers 1 --out "experiments/order03/$tag" \
        >> "logs/order03/${tag}_${ds}_${s}.log" 2>&1 &
    fi
  done < "$QUEUE"
  if [ "$pending" -eq 0 ]; then echo "QUEUE_EMPTY $(date)"; exit 0; fi
  sleep 20
done
