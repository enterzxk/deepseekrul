#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
nohup $PY scripts/run_femto_cqr.py --feat-dir data/femto_features --window 30 --epochs 120   --out experiments/femto > logs/femto_exp.log 2>&1 &
echo "PID=$!"
sleep 30
tail -5 logs/femto_exp.log 2>/dev/null || echo starting
