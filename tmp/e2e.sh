#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
nohup bash -c 'for S in FD001 FD002 FD003 FD004; do
  echo "=== $S ==="
  '"$PY"' scripts/run_e2e_cqr.py --subset $S --fold 0 --seed 42 --prior-dir experiments/ida/prior_boot --out experiments/punet_e2e 2>&1 | tail -2
done' > logs/e2e_cqr.log 2>&1 &
echo "PID=$!"
sleep 30
tail -3 logs/e2e_cqr.log
