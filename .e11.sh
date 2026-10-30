#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
: > /tmp/e11_status.txt
rm -f /tmp/e11_jobs.txt
for arch in cnn1d bilstm gru tcn tf; do
  for seed in 2027 2028 2029; do
    for fold in 0 1 2 3 4; do
      echo "$arch $seed $fold" >> /tmp/e11_jobs.txt
    done
  done
done
echo "queued $(wc -l < /tmp/e11_jobs.txt) at $(date -Is)" >> /tmp/e11_status.txt
cat /tmp/e11_jobs.txt | xargs -P 4 -n 3 bash -c '
  arch=$0; seed=$1; fold=$2
  out=experiments/e11/${arch}_s${seed}_f${fold}
  mkdir -p $out
  /home/yndx/miniconda3/envs/a40_env/bin/python -u scripts/run_candidate.py \
    --subset FD002 --variant $arch --seed $seed --split-seed 7 --fold $fold \
    --ref-mode global --epochs 80 --patience 20 \
    --prereg-sha 1e63c8001f9fcc4c95fe389a8078d87633b18bec8f1b4388916fffe4288b994e --out $out > /tmp/e11_${arch}_${seed}_f${fold}.log 2>&1
  echo "done $arch $seed f$fold rc=$? at $(date -Is)" >> /tmp/e11_status.txt
'
echo ALLDONE >> /tmp/e11_status.txt
