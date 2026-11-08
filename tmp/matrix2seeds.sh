#!/bin/bash
pkill -9 -f run_e2e_cqr 2>/dev/null
pkill -9 -f matrix_clean.sh 2>/dev/null
sleep 3
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python

worker() {
  S=$1
  for SEED in 42 7; do
    for F in 0 1 2 3 4; do
      D=experiments/punet_matrix/$S/f${F}_s${SEED}
      [ -f "$D/preds_cqr.csv" ] || {
        echo "=== $S f$F s$SEED cqr ==="
        $PY scripts/run_e2e_cqr.py --subset $S --fold $F --seed $SEED --heads cqr           --dump-preds --save-ckpt --prior-dir experiments/ida/prior_boot           --out experiments/punet_matrix 2>&1 | tail -1
      }
      [ -f "$D/preds_point.csv" ] || {
        echo "=== $S f$F s$SEED point ==="
        $PY scripts/run_e2e_cqr.py --subset $S --fold $F --seed $SEED --heads point           --dump-preds --save-ckpt --prior-dir experiments/ida/prior_boot           --out experiments/punet_matrix 2>&1 | tail -1
      }
    done
  done
  echo "DONE $S"
}
for S in FD001 FD002 FD003 FD004; do worker $S & done
wait
echo ALL_DONE
