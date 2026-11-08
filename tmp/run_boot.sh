#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
nohup $PY scripts/build_prior_uncertainty.py --subsets FD001 FD002 FD003 FD004 --bootstrap 20   --out experiments/ida/prior_boot > logs/prior_boot.log 2>&1 &
echo "PID=$!"
sleep 8
tail -3 logs/prior_boot.log 2>/dev/null || echo "starting"
