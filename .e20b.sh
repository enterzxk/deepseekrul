#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: > /tmp/e20b_status.txt
rm -f /tmp/e20b_jobs.txt
for arm in A BE PE; do
  for s in 2027 2028 2029; do
    for f in 0 1 2 3 4; do echo "$arm $s $f" >> /tmp/e20b_jobs.txt; done
  done
done
echo "queued FD004 $(wc -l < /tmp/e20b_jobs.txt) at $(date -Is)" >> /tmp/e20b_status.txt
cat /tmp/e20b_jobs.txt | xargs -P 4 -n 3 bash -c '
  arm=$0; seed=$1; fold=$2
  /home/yndx/miniconda3/envs/a40_env/bin/python -u scripts/run_candidate.py \
    --subset FD004 --variant $arm --seed $seed --split-seed 7 --fold $fold \
    --ref-mode global --epochs 80 --patience 20 --d-model 64 --num-layers 1 \
    --prereg-sha 1959e9244ea8d14f9bdc71152e793e986604b489cd43352285e3c113fefe1feb --out experiments/e20/FD004/${arm}_s${seed}_f${fold} \
    > /tmp/e20_FD004_${arm}_${seed}_f${fold}.log 2>&1
  echo "done FD004 $arm $seed f$fold rc=$? $(date +%H:%M)" >> /tmp/e20b_status.txt
'
echo ALLDONE >> /tmp/e20b_status.txt
