#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
# 1) train point models with checkpoint saving (parallel per subset)
for S in FD001 FD002 FD003 FD004; do
  ( $PY scripts/run_e2e_cqr.py --subset $S --fold 0 --seed 42 --heads point --save-ckpt       --prior-dir experiments/ida/prior_boot --out experiments/punet_e2e > logs/pt_$S.log 2>&1 ) &
done
wait
echo "=== training done, running MC dropout ==="
for S in FD001 FD002 FD003 FD004; do
  ( $PY scripts/run_mcdropout.py --subset $S --fold 0 --seed 42       --prior-dir experiments/ida/prior_boot --root experiments/punet_e2e > logs/mcd_$S.log 2>&1 ) &
done
wait
echo ALL_DONE
