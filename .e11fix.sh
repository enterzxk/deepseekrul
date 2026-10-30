#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
: > /tmp/e11fix_status.txt
for spec in "cnn1d 2028 1" "cnn1d 2028 3" "cnn1d 2029 2" "cnn1d 2029 4" "bilstm 2027 2"; do
  set -- $spec
  arch=$1; seed=$2; fold=$3
  $PY -u scripts/run_candidate.py --subset FD002 --variant $arch --seed $seed \
    --split-seed 7 --fold $fold --ref-mode global --epochs 80 --patience 20 \
    --prereg-sha 1e63c8001f9fcc4c95fe389a8078d87633b18bec8f1b4388916fffe4288b994e --out experiments/e11/${arch}_s${seed}_f${fold} \
    > /tmp/e11_${arch}_${seed}_f${fold}.log 2>&1 &
done
wait
echo ALLDONE >> /tmp/e11fix_status.txt
