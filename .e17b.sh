#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: > /tmp/e17b_status.txt
rm -f /tmp/e17b_jobs.txt
for arch in tf cnn1d; do for f in 0 1 2 3 4; do echo "$arch $f" >> /tmp/e17b_jobs.txt; done; done
echo "queued $(wc -l < /tmp/e17b_jobs.txt) at $(date -Is)" >> /tmp/e17b_status.txt
cat /tmp/e17b_jobs.txt | xargs -P 3 -n 2 bash -c '
  arch=$0; fold=$1
  /home/yndx/miniconda3/envs/a40_env/bin/python -u scripts/run_candidate.py \
    --subset FD002 --variant $arch --seed 2027 --split-seed 7 --fold $fold \
    --ref-mode global --epochs 80 --patience 20 --baseline-d-model 128 \
    --prereg-sha 8335783d2a96c9cd628973aa88126c2d7fecaf56bbbaf396ea1e2aefb6788c24 --out experiments/e17b/${arch}w128_s2027_f${fold} \
    > /tmp/e17b_${arch}_f${fold}.log 2>&1
  echo "done $arch f$fold rc=$? $(date +%H:%M)" >> /tmp/e17b_status.txt
'
echo ALLDONE >> /tmp/e17b_status.txt
