#!/bin/bash
set -u
cd /home/user/rulb/tim_20261105/full_20261105
mkdir -p logs
run_one() {
  gpu=$1; subset=$2; fold=$3; win=$4; tag=$5; l1=$6; l2=$7; l3=$8; tau=$9
  echo "start $subset $fold w$win $tag gpu$gpu $(date '+%F %T')" >> logs/v3d.log
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=6 /home/user/miniconda3/envs/pmv/bin/python run_v3b.py $subset $fold --arms M0,FULL --window $win --tag ${tag}_w${win} --mixup 0.2 >> logs/v3d_${subset}_${fold}_w${win}.log 2>&1
  echo "done $subset $fold w$win rc=$? $(date '+%F %T')" >> logs/v3d.log
}
export -f run_one
cat tasks_v3d.txt | xargs -P 2 -L 1 bash -c 'run_one $0 $1 $2 $3 $4 $5 $6 $7 $8'
echo "V3D COMPLETE $(date '+%F %T')" >> logs/v3d.log
