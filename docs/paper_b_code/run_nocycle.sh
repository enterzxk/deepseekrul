#!/bin/bash
cd /home/user/rulb
PY=/home/user/miniconda3/envs/pmv/bin/python
run_one() {
  local sub=$1 span=$2 gpu=$3
  local tag=NC_${sub}_L${span}
  if [ -f experiments/${tag}/${sub}/dspt_s101.csv ]; then echo "skip $tag"; return; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_dspt.py $sub 0 --seed 101 --mode long \
      --span $span --rul-cap 125 --epochs 40 --no-cycle --tag $tag > logs/${tag}.log 2>&1
  echo "done $tag rc=$? $(date '+%F %T')"
}
i=0
for span in 70 100; do
  for sub in FD001 FD004; do
    run_one $sub $span $(( i % 2 )) &
    i=$((i+1))
    if [ $(( i % 2 )) -eq 0 ]; then wait; fi
  done
done
wait
echo "NOCYCLE_DONE n=$i $(date '+%F %T')"