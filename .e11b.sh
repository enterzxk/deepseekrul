#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
: > /tmp/e11b_status.txt
rm -f /tmp/e11b_jobs.txt
for arch in tcn tf; do
  for s in 2027 2028 2029; do
    for f in 0 1 2 3 4; do
      d=experiments/e11/${arch}_s${s}_f${f}/FD002/${arch}_s${s}/metrics.json
      [ -f "$d" ] || echo "$arch $s $f" >> /tmp/e11b_jobs.txt
    done
  done
done
echo "queued $(wc -l < /tmp/e11b_jobs.txt)" >> /tmp/e11b_status.txt
cat /tmp/e11b_jobs.txt | xargs -P 3 -n 3 bash -c '
  arch=$0; seed=$1; fold=$2
  /home/yndx/miniconda3/envs/a40_env/bin/python -u scripts/run_candidate.py \
    --subset FD002 --variant $arch --seed $seed --split-seed 7 --fold $fold \
    --ref-mode global --epochs 80 --patience 20 --prereg-sha 1e63c8001f9fcc4c95fe389a8078d87633b18bec8f1b4388916fffe4288b994e \
    --out experiments/e11/${arch}_s${seed}_f${fold} \
    > /tmp/e11_${arch}_${seed}_f${fold}.log 2>&1
  echo "done $arch $seed f$fold rc=$? $(date +%H:%M)" >> /tmp/e11b_status.txt
'
echo ALLDONE >> /tmp/e11b_status.txt
