#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:/
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
rm -f /tmp/e3c_status.txt
for seed in 2027 2028 2029; do
  for sub in FD002 FD001; do
    out=experiments/e3c/PS_${sub}_s${seed}
    mkdir -p $out
    nohup $PY -u scripts/run_candidate.py --subset $sub --variant PS --seed $seed       --split-seed 7 --ref-mode global --epochs 80 --patience 20 --d-model 64 --num-layers 1       --prereg-sha 3cceb805530e843087fee60701f5a53abff86916af399c6d057496f9fd6ad9cb --out $out > /tmp/e3c_${sub}_${seed}.log 2>&1 &
  done
done
wait
echo ALLDONE >> /tmp/e3c_status.txt
