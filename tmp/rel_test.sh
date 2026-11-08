#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for S in FD003 FD004 FD001; do
  echo "=== $S engine_rel ==="
  $PY scripts/run_punet_stage.py --subset $S --fold 0 --seed 42 --stage heads --reuse-ckpt     --prior-dir experiments/ida/prior_boot --out experiments/punet_5fold --sigma-mode engine_rel 2>&1 | tail -1
done
