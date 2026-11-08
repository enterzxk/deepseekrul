#!/bin/bash
set -u
cd /home/user/rulb/tim_20261105/full_20261105
mkdir -p logs
echo "start w150 $(date +%F\ %T)" >> logs/w150.log
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 /home/user/miniconda3/envs/pmv/bin/python run_full_fold3.py FD002 2 --seed 101 --arms M0,FULL --window 150 --tag w150 >> logs/FD002_f2_w150.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6 /home/user/miniconda3/envs/pmv/bin/python run_full_fold3.py FD002 2 --seed 101 --arms M0,FULL --window 100 --tag w100 >> logs/FD002_f2_w100.log 2>&1 &
P2=$!
wait $P1 $P2
echo "done w150/w100 $(date +%F\ %T)" >> logs/w150.log
