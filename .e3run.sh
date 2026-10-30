#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
rm -f /tmp/e3_status.txt
for arm in A BE PE; do
  for seed in 2027 2028 2029; do
    out=experiments/e3/${arm}_s${seed}
    mkdir -p $out
    nohup $PY -u scripts/run_candidate.py --subset FD002 --variant $arm --seed $seed \
      --split-seed 7 --ref-mode global --epochs 80 --patience 20 --d-model 64 --num-layers 1 \
      --out $out > /tmp/e3_${arm}_${seed}.log 2>&1 &
  done
  wait
  echo "batch $arm done" >> /tmp/e3_status.txt
done
echo ALLDONE >> /tmp/e3_status.txt
