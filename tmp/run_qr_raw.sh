#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for S in FD001 FD002 FD003 FD004; do
  ( $PY scripts/run_e2e_cqr.py --subset $S --fold 0 --seed 42 --heads qr_raw       --prior-dir experiments/ida/prior_boot --out experiments/punet_e2e > logs/qrraw_$S.log 2>&1 ) &
done
wait
echo DONE
