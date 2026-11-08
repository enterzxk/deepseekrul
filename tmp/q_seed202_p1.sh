#!/bin/bash
set -u
cd /home/user/rulb/tim_20261105/full_20261105
mkdir -p logs
run_one() {
  gpu=$1; subset=$2; fold=$3; seed=$4; win=$5
  echo "start $subset $fold s$seed w$win gpu$gpu $(date '+%F %T')" >> logs/seed202.log
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=5 /home/user/miniconda3/envs/pmv/bin/python run_full_fold3.py $subset $fold --seed $seed --arms M0,FULL --window $win --tag w$win >> logs/${subset}_f${fold}_s${seed}_w${win}.log 2>&1
  echo "done $subset $fold s$seed w$win rc=$? $(date '+%F %T')" >> logs/seed202.log
}
export -f run_one
cat tasks_seed202.txt | xargs -P 1 -L 1 bash -c 'run_one $0 $1 $2 $3 $4'
echo "SEED202 COMPLETE $(date '+%F %T')" >> logs/seed202.log
