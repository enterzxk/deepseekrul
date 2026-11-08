#!/bin/bash
# Efficient matrix: train once per (subset, fold, seed) and dump predictions.
# All calibration variants are derived offline from the dumps.
pkill -9 -f "run_e2e_cqr.py --subset" 2>/dev/null
sleep 2
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python

run_subset() {
  S=$1
  for SEED in 42 7 2024; do
    for F in 0 1 2 3 4; do
      echo "=== $S f$F s$SEED quantile ==="
      $PY scripts/run_e2e_cqr.py --subset $S --fold $F --seed $SEED --heads cqr         --dump-preds --save-ckpt --prior-dir experiments/ida/prior_boot         --out experiments/punet_matrix 2>&1 | tail -1
      echo "=== $S f$F s$SEED point ==="
      $PY scripts/run_e2e_cqr.py --subset $S --fold $F --seed $SEED --heads point         --dump-preds --save-ckpt --prior-dir experiments/ida/prior_boot         --out experiments/punet_matrix 2>&1 | tail -1
    done
  done
  echo "DONE $S"
}
for S in FD001 FD002 FD003 FD004; do run_subset $S & done
wait
echo ALL_DONE
