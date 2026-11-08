#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for S in FD002 FD001; do
  echo "=== $S weighted ==="
  $PY scripts/run_punet_stage.py --subset $S --fold 0 --seed 42 --stage heads --reuse-ckpt     --prior-dir experiments/ida/prior_boot --out experiments/punet_boot --calib-mode weighted 2>&1 | tail -1
done
