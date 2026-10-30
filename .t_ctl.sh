#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for v in Anc Cyc; do
  for s in 2027 2028; do
    ( $PY -u scripts/run_candidate.py --subset FD001 --variant $v --seed $s --epochs 15 --patience 8         --d-model 64 --num-layers 1 --out experiments/tournament > logs/tournament/FD001_${v}_${s}.log 2>&1 ) &
  done
done
wait
echo CONTROLS_DONE
