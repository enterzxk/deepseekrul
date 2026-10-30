#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:/
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
rm -f /tmp/e3b_status.txt
for arm in A BE PE; do
  for seed in 2027 2028 2029; do
    out=experiments/e3_fd001/${arm}_s${seed}
    mkdir -p $out
    nohup $PY -u scripts/run_candidate.py --subset FD001 --variant $arm --seed $seed       --split-seed 7 --ref-mode global --epochs 80 --patience 20 --d-model 64 --num-layers 1       --prereg-sha 3c2b8ea0235d1e35957aff52e041aeac4178e3bd082d9adc1b26eaa8a4d4e3d7 --out $out > /tmp/e3b_${arm}_${seed}.log 2>&1 &
  done
  wait
  echo "batch $arm done" >> /tmp/e3b_status.txt
done
echo ALLDONE >> /tmp/e3b_status.txt
