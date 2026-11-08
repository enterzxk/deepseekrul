#!/bin/bash
pkill -9 -f "run_e2e_cqr.py --subset" 2>/dev/null
sleep 2
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python

worker() {
  S=$1; FOLDS=$2
  for SEED in 42 7 2024; do
    for F in $FOLDS; do
      D=experiments/punet_matrix/$S/f${F}_s${SEED}
      if [ -f "$D/preds_cqr.csv" ] && [ -f "$D/preds_point.csv" ]; then
        echo "skip $S f$F s$SEED"
        continue
      fi
      echo "=== $S f$F s$SEED quantile ==="
      $PY scripts/run_e2e_cqr.py --subset $S --fold $F --seed $SEED --heads cqr         --dump-preds --save-ckpt --prior-dir experiments/ida/prior_boot         --out experiments/punet_matrix 2>&1 | tail -1
      echo "=== $S f$F s$SEED point ==="
      $PY scripts/run_e2e_cqr.py --subset $S --fold $F --seed $SEED --heads point         --dump-preds --save-ckpt --prior-dir experiments/ida/prior_boot         --out experiments/punet_matrix 2>&1 | tail -1
    done
  done
  echo "DONE $S $FOLDS"
}

worker FD001 "0 1 2" & worker FD001 "3 4" &
worker FD002 "0 1 2" & worker FD002 "3 4" &
worker FD003 "0 1 2" & worker FD003 "3 4" &
worker FD004 "0 1 2" & worker FD004 "3 4" &
wait
echo ALL_DONE
