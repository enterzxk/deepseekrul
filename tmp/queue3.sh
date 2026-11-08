#!/bin/bash
set -u
cd /home/user/rulb/tim_20261105/full_20261105
mkdir -p logs
run_one() {
  gpu=$1; subset=$2; fold=$3; arms=$4
  echo "start $subset $fold gpu$gpu $(date '+%F %T')" >> logs/q3.log
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=6 /home/user/miniconda3/envs/pmv/bin/python run_full_fold2.py $subset $fold --seed 101 --arms $arms >> logs/v2_${subset}_f${fold}.log 2>&1
  echo "done $subset $fold rc=$? $(date '+%F %T')" >> logs/q3.log
  for cond in clean mc1 mc2; do
    OMP_NUM_THREADS=4 /home/user/miniconda3/envs/pmv/bin/python measurement_conditions.py $subset $fold M0 101 "$cond" --rundir runs2 >> logs/mc_${subset}_f${fold}.log 2>&1
    OMP_NUM_THREADS=4 /home/user/miniconda3/envs/pmv/bin/python measurement_conditions.py $subset $fold FULL 101 "$cond" --rundir runs2 >> logs/mc_${subset}_f${fold}.log 2>&1
  done
  echo "mc done $subset $fold $(date '+%F %T')" >> logs/q3.log
}
export -f run_one
cat tasks_q3.txt | xargs -P 2 -L 1 bash -c 'run_one $0 $1 $2 $3'
echo "Q3 COMPLETE $(date '+%F %T')" >> logs/q3.log
