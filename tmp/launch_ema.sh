#!/bin/bash
cd /home/user/rulb
PY=/home/user/miniconda3/envs/pmv/bin/python
run_one() {
  local sub=$1 seed=$2 fold=$3 gpu=$4
  local name=EMA_${sub}_s${seed}_f${fold}
  if [ -f experiments/${name}/${sub}/dspt_s${seed}.csv ]; then echo "skip $name"; return; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_dspt.py $sub $fold --seed $seed --mode long --tag $name > logs/${name}.log 2>&1
  echo "done $name rc=$? $(date '+%F %T')"
}
i=0
for sub in FD001 FD003; do
  for fold in 0 1 2 3 4; do
    run_one $sub 101 $fold $(( i % 2 )) &
    i=$((i+1))
    if [ $(( i % 2 )) -eq 0 ]; then wait; fi
  done
done
wait
echo ALLDONE $(date '+%F %T')