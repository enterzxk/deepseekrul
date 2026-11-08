#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
mkdir -p logs
nohup bash -c 'for S in FD001 FD002 FD003 FD004; do
  echo "=== START $S $(date +%H:%M:%S) ==="
  '"$PY"' scripts/run_ida.py --subset $S --arm M1 --fold 0 --seed 42 --epochs 80 --patience 20 2>&1
  echo "=== END $S $(date +%H:%M:%S) rc=$? ==="
done' > logs/ida_m1_f0.log 2>&1 &
echo "PID=$!"
sleep 25
tail -6 logs/ida_m1_f0.log 2>/dev/null || echo "no output yet"
