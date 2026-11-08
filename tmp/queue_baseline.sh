#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
# wait for the running 5-fold CQR job to finish (no GPU contention)
while kill -0 2408133 2>/dev/null; do sleep 60; done
nohup bash -c '
for S in FD001 FD002 FD003 FD004; do
  for F in 0 1 2 3 4; do
    echo "=== $S f$F point ==="
    '"$PY"' scripts/run_e2e_cqr.py --subset $S --fold $F --seed 42 --heads point       --prior-dir experiments/ida/prior_boot --out experiments/punet_e2e 2>&1 | tail -1
  done
done' > logs/e2e_point_5fold.log 2>&1 &
echo "QUEUED PID=$!"
