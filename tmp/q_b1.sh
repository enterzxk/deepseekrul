#!/bin/bash
cd /home/user/rulb
PY=/home/user/miniconda3/envs/pmv/bin/python
for spec in "FD002 2" "FD002 3" "FD004 2" "FD004 3"; do
  set -- $spec
  tag="B1_${1}_slot_f${2}"
  echo "start $tag $(date '+%F %T')" >> logs/b1.log
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 $PY scripts/run_dspt.py $1 $2 \
    --mode slot --epochs 80 --patience 20 --seed 101 --tag $tag >> logs/${tag}.log 2>&1
  echo "done $tag rc=$? $(date '+%F %T')" >> logs/b1.log
done
echo "B1 COMPLETE $(date '+%F %T')" >> logs/b1.log
