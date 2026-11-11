#!/bin/bash
cd /home/user/rulb
PY=/home/user/miniconda3/envs/pmv/bin/python
run_one() {
  local span=$1 gpu=$2
  local full=MSW_FD003_L${span}
  if [ -f experiments/${full}/FD003/dspt_s101.csv ]; then echo "skip $full"; return; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_dspt.py FD003 0 --seed 101 --mode long \
      --span $span --rul-cap 125 --epochs 40 --hi-lambda 0.5 --hi-mono 0.1 \
      --tag $full > logs/${full}.log 2>&1
  echo "done $full rc=$? $(date '+%F %T')"
}
i=0
for span in 30 50 70 100; do
  run_one $span $(( i % 2 )) &
  i=$((i+1))
  if [ $(( i % 2 )) -eq 0 ]; then wait; fi
done
wait
echo "FD003_DONE n=$i $(date '+%F %T')"