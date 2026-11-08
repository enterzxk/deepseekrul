#!/bin/bash
cd /home/user/rulb
PY=/home/user/miniconda3/envs/pmv/bin/python
while read -r tag sub seed fold; do
  [ -z "$tag" ] && continue
  gpu=$(( fold % 2 ))
  name=L2_${sub}_s${seed}_f${fold}
  if [ -f experiments/${name}/${sub}/dspt_s${seed}.csv ]; then echo "skip $name"; continue; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_dspt.py --subset $sub --fold $fold --seed $seed --mode long --layers 2 --tag $name > logs/${name}.log 2>&1
  echo "done $name rc=$? $(date '+%F %T')"
done < tasks_l2.txt