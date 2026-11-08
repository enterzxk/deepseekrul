#!/bin/bash
pkill -9 -f run_femto_cqr 2>/dev/null
sleep 2
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
nohup $PY scripts/run_femto_cqr.py --feat-dir data/femto_features --window 30 --epochs 30   --max-windows 400 --out experiments/femto > logs/femto_fast.log 2>&1 &
echo "PID=$!"
sleep 60
tail -3 logs/femto_fast.log | cut -c1-70
ls experiments/femto/*/preds_q.csv 2>/dev/null | wc -l
