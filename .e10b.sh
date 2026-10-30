#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
: > /tmp/e10b_status.txt
for fold in 1 2 3 4; do
  for seed in 2027 2028 2029; do
    echo "BE $seed $fold"
  done
done > /tmp/e10b_jobs.txt
cat /tmp/e10b_jobs.txt | xargs -P 4 -n 3 bash -c '
  arm=$0; seed=$1; fold=$2
  out=experiments/e10b/${arm}_s${seed}_f${fold}
  mkdir -p $out
  /home/yndx/miniconda3/envs/a40_env/bin/python -u scripts/run_candidate.py \
    --subset FD002 --variant $arm --seed $seed --split-seed 7 --fold $fold \
    --ref-mode global --epochs 80 --patience 20 --d-model 64 --num-layers 1 \
    --prereg-sha 532a1d5de4aaa10d1e05f8a45f92697a080c8da34cdeff337bb13b5be0ec7cd1 --out $out > /tmp/e10b_${arm}_${seed}_f${fold}.log 2>&1
  echo "done $arm $seed f$fold rc=$? at $(date -Is)" >> /tmp/e10b_status.txt
'
echo ALLDONE >> /tmp/e10b_status.txt
