#!/bin/bash
set -u
cd /home/user/rulb/tim_20261105/full_20261105
mkdir -p logs
run_one() {
  gpu=$1; subset=$2; fold=$3; seed=$4; arms=$5
  log=logs/${subset}_f${fold}_s${seed}.log
  echo "start $subset $fold $seed gpu$gpu $(date '+%F %T')" >> logs/q2.log
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=6 /home/user/miniconda3/envs/pmv/bin/python run_full_fold.py $subset $fold --seed $seed --arms $arms >> $log 2>&1
  echo "done $subset $fold $seed rc=$? $(date '+%F %T')" >> logs/q2.log
}
export -f run_one
cat tasks_q2.txt | xargs -P 2 -L 1 bash -c 'run_one $0 $1 $2 $3 $4'
echo "Q2 COMPLETE $(date '+%F %T')" >> logs/q2.log
