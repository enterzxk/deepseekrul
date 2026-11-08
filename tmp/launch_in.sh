#!/bin/bash
cd /home/user/rulb
PY=/home/user/miniconda3/envs/pmv/bin/python
run_one() {
  local sub=$1 seed=$2 fold=$3 gpu=$4
  local name=IN_${sub}_s${seed}_f${fold}
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_dspt.py $sub $fold --seed $seed --mode long --instnorm --tag $name > logs/${name}.log 2>&1
  echo "done $name rc=$? $(date '+%F %T')"
}
run_one FD003 101 0 0 &
run_one FD003 101 1 1 &
wait
echo ALLDONE $(date '+%F %T')