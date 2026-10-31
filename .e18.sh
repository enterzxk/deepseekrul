#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: > /tmp/e18_status.txt
rm -f /tmp/e18_jobs.txt
for arm in A BE PE; do
  for s in 2027 2028 2029; do
    for f in 1 2 3 4; do echo "$arm $s $f" >> /tmp/e18_jobs.txt; done
  done
done
echo "queued $(wc -l < /tmp/e18_jobs.txt) at $(date -Is)" >> /tmp/e18_status.txt
cat /tmp/e18_jobs.txt | xargs -P 4 -n 3 bash -c '
  arm=$0; seed=$1; fold=$2
  /home/yndx/miniconda3/envs/a40_env/bin/python -u scripts/run_candidate.py \
    --subset FD001 --variant $arm --seed $seed --split-seed 7 --fold $fold \
    --ref-mode global --epochs 80 --patience 20 --d-model 64 --num-layers 1 \
    --prereg-sha 59bb5d9bf0870c006cbf262eb6375d5634abbcb69b1b72c159d024e6935f4ffa --out experiments/e18/${arm}_s${seed}_f${fold} \
    > /tmp/e18_${arm}_${seed}_f${fold}.log 2>&1
  echo "done $arm $seed f$fold rc=$? $(date +%H:%M)" >> /tmp/e18_status.txt
'
echo ALLDONE >> /tmp/e18_status.txt
