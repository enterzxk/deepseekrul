#!/bin/bash
cd /home/user/rulb/tim_20261105/full_20261105
PY=/home/user/miniconda3/envs/pmv/bin/python
# C1: small width + modules, strong regularisation
setsid nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 $PY run_deep2.py FD002 2 --arms M0 --d-model 96 --layers 1 --extra-layers 0 --dropout 0.3 --wd 1e-3 --tag c1 > logs/sweep_c1.log 2>&1 < /dev/null &
sleep 2
# C2: deep + strong regularisation
setsid nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 $PY run_deep2.py FD002 2 --arms M0 --d-model 128 --layers 2 --extra-layers 1 --dropout 0.3 --wd 1e-3 --tag c2 > logs/sweep_c2.log 2>&1 < /dev/null &
sleep 40
tail -n 3 logs/sweep_c1.log; echo ---; tail -n 3 logs/sweep_c2.log; date
