#!/bin/bash
cd /home/user/rulb
PY=/home/user/miniconda3/envs/pmv/bin/python
run_one() {
  local sub=$1 tag=$2 nc=$3 gpu=$4
  local full=HI_${sub}_${tag}
  if [ -f experiments/${full}/${sub}/dspt_s101.csv ]; then echo "skip $full"; return; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_dspt.py $sub 0 --seed 101 --mode long \
      --span 70 --rul-cap 125 --epochs 40 --hi-lambda 0.5 --hi-mono 0.1 $nc --tag $full \
      > logs/${full}.log 2>&1
  echo "done $full rc=$? $(date '+%F %T')"
}
i=0
for sub in FD001 FD004; do
  run_one $sub CYC  ''         $(( i % 2 )) &
  i=$((i+1)); if [ $(( i % 2 )) -eq 0 ]; then wait; fi
  run_one $sub NOCYC '--no-cycle' $(( i % 2 )) &
  i=$((i+1)); if [ $(( i % 2 )) -eq 0 ]; then wait; fi
done
wait
echo "HI_DONE n=$i $(date '+%F %T')"