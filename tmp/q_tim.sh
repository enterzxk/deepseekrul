#!/bin/bash
set -u
cd /home/user/rulb
mkdir -p logs
run_one() {
  gpu=$1; subset=$2; fold=$3; win=$4; mix=$5
  tag="${subset}_w${win}_m${mix}_f${fold}"
  echo "start $tag gpu$gpu $(date '+%F %T')" >> logs/tim_test.log
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=6 /home/user/miniconda3/envs/pmv/bin/python scripts/run_candidate.py \
    --subset $subset --variant PE --seed 2027 --fold $fold --window $win --mixup $mix \
    --epochs 40 --out experiments/tim_$tag >> logs/tim_$tag.log 2>&1
  echo "done $tag rc=$? $(date '+%F %T')" >> logs/tim_test.log
}
export -f run_one
cat tasks_tim.txt | xargs -P 2 -L 1 bash -c 'run_one $0 $1 $2 $3 $4'
echo "TIM TEST COMPLETE $(date '+%F %T')" >> logs/tim_test.log
