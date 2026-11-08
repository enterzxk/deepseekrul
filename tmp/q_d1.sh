#!/bin/bash
set -u
cd /home/user/rulb
mkdir -p logs
run_one() {
  gpu=$1; subset=$2; fold=$3; mode=$4
  tag="D1_${subset}_${mode}_f${fold}"
  echo "start $tag gpu$gpu $(date '+%F %T')" >> logs/d1.log
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=6 /home/user/miniconda3/envs/pmv/bin/python scripts/run_dspt.py \
    $subset $fold --mode $mode --epochs 80 --patience 20 --seed 101 --tag $tag >> logs/${tag}.log 2>&1
  echo "done $tag rc=$? $(date '+%F %T')" >> logs/d1.log
}
export -f run_one
cat tasks_d1.txt | xargs -P 2 -L 1 bash -c 'run_one $0 $1 $2 $3'
echo "D1 COMPLETE $(date '+%F %T')" >> logs/d1.log
