#!/bin/bash
S=$1
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for F in 0 1 2 3 4; do
  echo "=== $S f$F point ==="
  $PY scripts/run_e2e_cqr.py --subset $S --fold $F --seed 42 --heads point     --prior-dir experiments/ida/prior_boot --out experiments/punet_e2e 2>&1 | tail -1
done
echo "DONE $S"
