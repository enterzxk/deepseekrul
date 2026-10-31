#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
: > /tmp/e11c_status.txt
for arch in tcn tf; do
  for f in 0 1 2 3 4; do
    echo "$arch 2027 $f"
  done
done > /tmp/e11c_jobs.txt
cat /tmp/e11c_jobs.txt | xargs -P 3 -n 3 bash -c '
  arch=$0; seed=$1; fold=$2
  /home/yndx/miniconda3/envs/a40_env/bin/python -u scripts/run_candidate.py \
    --subset FD002 --variant $arch --seed $seed --split-seed 7 --fold $fold \
    --ref-mode global --epochs 80 --patience 20 --prereg-sha b8eeb75751b25238029b716d1ec32802838eeffe9ef62f027908081a97570cc2 \
    --out experiments/e11/${arch}_s${seed}_f${fold} \
    > /tmp/e11_${arch}_${seed}_f${fold}.log 2>&1
  echo "done $arch f$fold rc=$? $(date +%H:%M)" >> /tmp/e11c_status.txt
'
echo ALLDONE >> /tmp/e11c_status.txt
