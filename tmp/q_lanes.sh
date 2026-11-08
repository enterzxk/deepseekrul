#!/bin/bash
# GPU-pinned scheduler: two lanes, each runs its own task list sequentially.
# usage: bash q_lanes.sh <tasks.txt> <logfile> <tagprefix>
set -u
TASKS=$1
LOG=$2
cd /home/user/rulb
PY=/home/user/miniconda3/envs/pmv/bin/python

lane() {
  local gpu=$1
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    set -- $line
    [ "$1" != "$gpu" ] && continue
    local subset=$2 fold=$3 mode=$4 tag=$5
    echo "start $tag gpu$gpu $(date '+%F %T')" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=6 $PY scripts/run_dspt.py \
      $subset $fold --mode $mode --epochs 80 --patience 20 --seed 101 --tag $tag \
      >> logs/${tag}.log 2>&1
    echo "done $tag rc=$? $(date '+%F %T')" >> "$LOG"
  done < "$TASKS"
}

lane 0 &
lane 1 &
wait
echo "ALL DONE $(date '+%F %T')" >> "$LOG"
