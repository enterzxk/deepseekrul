#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
: > /tmp/e17_status.txt
rm -f /tmp/e17_jobs.txt
for arch in tf cnn1d; do
  for f in 0 1 2 3 4; do echo "$arch $f" >> /tmp/e17_jobs.txt; done
done
echo "queued $(wc -l < /tmp/e17_jobs.txt) at $(date -Is)" >> /tmp/e17_status.txt
cat /tmp/e17_jobs.txt | xargs -P 3 -n 2 bash -c '
  arch=$0; fold=$1
  /home/yndx/miniconda3/envs/a40_env/bin/python -u scripts/run_candidate.py \
    --subset FD002 --variant $arch --seed 2027 --split-seed 7 --fold $fold \
    --ref-mode global --epochs 80 --patience 20 --d-model 128 --num-layers 2 \
    --prereg-sha c943daf43bcde936f7b785333570d31a2b6c7ecd82c9d372542aaf620324b2c5 --out experiments/e17/${arch}big_s2027_f${fold} \
    > /tmp/e17_${arch}_f${fold}.log 2>&1
  echo "done $arch f$fold rc=$? $(date +%H:%M)" >> /tmp/e17_status.txt
'
echo ALLDONE >> /tmp/e17_status.txt
