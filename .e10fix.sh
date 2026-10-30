#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
: > /tmp/e10fix_status.txt
run () {
  local arm=$1 seed=$2 fold=$3
  $PY -u scripts/run_candidate.py --subset FD002 --variant $arm --seed $seed \
    --split-seed 7 --fold $fold --ref-mode global --epochs 80 --patience 20 \
    --d-model 64 --num-layers 1 --prereg-sha aa7072f745c9ea2f552fb7e3203d1b14ef1501f62d01e445a919329adc1453b4 \
    --out experiments/e10/${arm}_s${seed}_f${fold} > /tmp/e10_${arm}_${seed}_f${fold}.log 2>&1
  echo "done $arm $seed f$fold rc=$? at $(date -Is)" >> /tmp/e10fix_status.txt
}
run PE 2028 1 &
run PE 2029 4 &
wait
echo ALLDONE >> /tmp/e10fix_status.txt
