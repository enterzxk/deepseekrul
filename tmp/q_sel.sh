#!/bin/bash
set -u
cd /home/user/rulb/tim_20261105/full_20261105
mkdir -p logs
run_one() {
  gpu=$1; fold=$2; mix=$3
  echo "start f$fold m$mix gpu$gpu $(date '+%F %T')" >> logs/select.log
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=6 /home/user/miniconda3/envs/pmv/bin/python run_v3b.py FD002 $fold \
    --arms M0 --window 50 --mixup $mix --select best --tag SEL_best_m$mix >> logs/sel_f${fold}_m${mix}.log 2>&1
  echo "done f$fold m$mix rc=$? $(date '+%F %T')" >> logs/select.log
}
export -f run_one
cat tasks_sel.txt | xargs -P 2 -L 1 bash -c 'run_one $0 $1 $2'
echo "SELECT COMPLETE $(date '+%F %T')" >> logs/select.log
