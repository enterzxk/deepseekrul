#!/bin/bash
set -e
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
mkdir -p experiments/ida/prior logs
nohup $PY src/lifetime_prior/nested_prior.py   --data data/CMAPSS   --subsets FD001 FD002 FD003 FD004   --out experiments/ida/prior   > logs/prior_gen.log 2>&1 &
echo "PID=$!"
sleep 20
tail -6 logs/prior_gen.log || true
