#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
nohup bash -c '
for F in 0 1 2 3 4; do
  echo "=== FD004 f$F ema ==="
  '"$PY"' scripts/run_e2e_cqr.py --subset FD004 --fold $F --seed 42 --heads cqr     --prior-dir experiments/ida/prior_boot --out experiments/punet_e2e_ema 2>&1 | tail -1
done' > logs/fd004_ema.log 2>&1 &
echo "PID=$!"
sleep 8
tail -2 logs/fd004_ema.log 2>/dev/null || echo starting
