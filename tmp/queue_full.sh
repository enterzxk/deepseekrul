#!/bin/bash
set -u
cd /home/user/rulb/tim_20261105/full_20261105
mkdir -p logs
run_one() {
  gpu=$1; subset=$2; fold=$3; arms=$4
  log=logs/$subset"_f"$fold"_s101.log"
  if [ -f runs/$subset/$fold/done_101.json ]; then echo "skip $subset $fold"; return; fi
  echo "start $subset $fold gpu$gpu $(date '+%F %T')" >> logs/queue.log
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 /home/user/miniconda3/envs/pmv/bin/python run_full_fold.py $subset $fold --seed 101 --arms $arms >> $log 2>&1
  echo "done $subset $fold rc=$? $(date '+%F %T')" >> logs/queue.log
}
export -f run_one
awk '{print $1" "$2" "$3" "$4}' tasks.txt | xargs -P 4 -L 1 bash -c 'run_one $0 $1 $2 $3'
echo "QUEUE COMPLETE $(date '+%F %T')" >> logs/queue.log
