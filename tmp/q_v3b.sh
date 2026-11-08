#!/bin/bash
set -u
cd /home/user/rulb/tim_20261105/full_20261105
mkdir -p logs
run_one() {
  gpu=$1; subset=$2; fold=$3; tag=$4; l1=$5; l2=$6; l3=$7; tau=$8; mode=$9; mix=${10}
  echo "start $subset $fold $tag gpu$gpu $(date '+%F %T')" >> logs/v3b.log
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=6 /home/user/miniconda3/envs/pmv/bin/python run_v3b.py $subset $fold --arms M0,FULL --window 50 --tag $tag --l1 $l1 --l2 $l2 --l3 $l3 --tau $tau --target-mode $mode --mixup $mix >> logs/v3b_${subset}_${fold}_${tag}.log 2>&1
  echo "done $subset $fold $tag rc=$? $(date '+%F %T')" >> logs/v3b.log
}
export -f run_one
cat tasks_v3b.txt | xargs -P 2 -L 1 bash -c 'run_one $0 $1 $2 $3 $4 $5 $6 $7 $8 $9'
echo "V3B COMPLETE $(date '+%F %T')" >> logs/v3b.log
