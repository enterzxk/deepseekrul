#!/bin/bash
pkill -9 -f run_femto_cqr 2>/dev/null
sleep 2
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
nohup $PY scripts/run_femto_cqr.py --feat-dir data/femto_features --window 30 --epochs 40   --out experiments/femto > logs/femto_gpu.log 2>&1 &
echo "PID=$!"
sleep 40
tail -2 logs/femto_gpu.log
