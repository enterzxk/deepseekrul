#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for S in FD001 FD002 FD003 FD004; do
  for F in 0 1 2 3 4; do
    D=experiments/punet_matrix/$S/f${F}_s42
    [ -f "$D/model.pt" ] && [ -f "$D/preds_point.csv" ] && [ ! -f "$D/preds_mcd.csv" ] && {
      $PY scripts/run_mcdropout.py --subset $S --fold $F --seed 42         --prior-dir experiments/ida/prior_boot --root experiments/punet_matrix 2>&1 | tail -1
    }
  done
done
echo MCD_DONE
