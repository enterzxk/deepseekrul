#!/bin/bash
set -u
TASKS=/home/user/rulb/tasks_matrix.txt
LOG=/home/user/rulb/logs/matrix.log
cd /home/user/rulb
PY=/home/user/miniconda3/envs/pmv/bin/python
lane() {
  local gpu=$1
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    set -- $line
    [ "$1" != "$gpu" ] && continue
    local subset=$2 fold=$3 seed=$4 tag=$5
    echo "start $tag gpu$gpu $(date '+%F %T')" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=6 $PY scripts/run_dspt.py \
      $subset $fold --mode long --epochs 80 --patience 20 --seed $seed --tag $tag \
      >> logs/${tag}.log 2>&1
    echo "done $tag rc=$? $(date '+%F %T')" >> "$LOG"
  done < "$TASKS"
}
lane 0 &
lane 1 &
wait
echo "MATRIX COMPLETE $(date '+%F %T')" >> "$LOG"
