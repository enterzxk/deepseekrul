#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: > /tmp/e20f4_status.txt
rm -f /tmp/e20f4_jobs.txt
for arm in A BE PE; do
  for s in 2027 2028 2029; do
    if [ -f experiments/e20/FD004/${arm}_s${s}_f4/FD004/${arm}_s${s}/metrics.json ]; then continue; fi
    echo "$arm $s 4" >> /tmp/e20f4_jobs.txt
  done
done
echo "FD004 fold4 todo $(wc -l < /tmp/e20f4_jobs.txt) at $(date -Is)" >> /tmp/e20f4_status.txt
cat /tmp/e20f4_jobs.txt | xargs -P 2 -n 3 bash -c '
  arm=$0; seed=$1; fold=$2
  /home/yndx/miniconda3/envs/a40_env/bin/python -u scripts/run_candidate.py     --subset FD004 --variant $arm --seed $seed --split-seed 7 --fold $fold     --ref-mode global --epochs 80 --patience 20 --d-model 64 --num-layers 1     --prereg-sha 3f0142f96b8e2971 --out experiments/e20/FD004/${arm}_s${seed}_f${fold}     > /tmp/e20_FD004_${arm}_${seed}_f${fold}.log 2>&1
  echo "done FD004 $arm $seed f$fold rc=$? $(date +%H:%M)" >> /tmp/e20f4_status.txt
'
echo ALLDONE >> /tmp/e20f4_status.txt
