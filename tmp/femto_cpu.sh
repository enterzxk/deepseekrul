#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=""
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
nohup $PY scripts/run_femto_cqr.py --feat-dir data/femto_features --window 30 --epochs 40   --out experiments/femto > logs/femto_cpu.log 2>&1 &
echo "PID=$!"
sleep 30
tail -4 logs/femto_cpu.log 2>/dev/null || echo starting
