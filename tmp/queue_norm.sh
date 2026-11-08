#!/bin/bash
set -u
cd /home/user/rulb/tim_20261105/full_20261105
mkdir -p logs
run_norm() {
  gpu=$1; subset=$2; fold=$3
  if [ -f runs/$subset/$fold/NORM_101.csv ]; then echo "skip $subset $fold"; return; fi
  echo "start $subset $fold $(date '+%F %T')" >> logs/norm_queue.log
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=6 /home/user/miniconda3/envs/pmv/bin/python run_full_fold.py $subset $fold --seed 101 --arms NORM >> logs/${subset}_f${fold}_norm.log 2>&1
  echo "done $subset $fold rc=$? $(date '+%F %T')" >> logs/norm_queue.log
}
export -f run_norm
printf '%s\n' '0 FD002 0' '1 FD002 1' '0 FD002 2' '1 FD002 3' '0 FD002 4' > tasks_norm.txt
cat tasks_norm.txt | xargs -P 2 -L 1 bash -c 'run_norm $0 $1 $2'
echo "NORM QUEUE COMPLETE $(date '+%F %T')" >> logs/norm_queue.log
