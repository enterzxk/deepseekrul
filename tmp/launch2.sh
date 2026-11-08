#!/bin/bash
cd /home/user/rulb/tim_20261105/full_20261105
PY=/home/user/miniconda3/envs/pmv/bin/python
setsid nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 $PY run_v3.py FD004 2 --arms M0,FULL --window 50 --tag B --l3 0.1 --tau 0.9 > logs/x_B_fd004.log 2>&1 < /dev/null &
sleep 2
setsid nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 $PY run_v3b.py FD002 2 --arms M0,FULL --window 50 --tag C --target-mode life > logs/x_C_fd002.log 2>&1 < /dev/null &
sleep 3
echo LAUNCHED
