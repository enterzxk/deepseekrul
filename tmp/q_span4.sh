#!/bin/bash
set -u
cd /home/user/rulb/tim_20261105/full_20261105
mkdir -p logs
run_one() {
  gpu=$1; subset=$2; fold=$3; win=$4
  echo "start $subset $fold w$win gpu$gpu $(date '+%F %T')" >> logs/span4.log
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=5 /home/user/miniconda3/envs/pmv/bin/python run_full_fold3.py $subset $fold --seed 101 --arms M0,FULL --window $win --tag w$win >> logs/${subset}_f${fold}_w${win}.log 2>&1
  echo "done $subset $fold w$win rc=$? $(date '+%F %T')" >> logs/span4.log
}
export -f run_one
cat tasks_span4.txt | xargs -P 2 -L 1 bash -c 'run_one $0 $1 $2 $3'
echo "SPAN4 COMPLETE $(date '+%F %T')" >> logs/span4.log
