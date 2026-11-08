#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
run_subset() {
  local S=$1
  for F in 0 1 2 3 4; do
    echo "=== $S f$F A4 ==="
    $PY scripts/run_e2e_cqr.py --subset $S --fold $F --seed 42 --heads cqr --ablate A4       --prior-dir experiments/ida/prior_boot --out experiments/punet_e2e 2>&1 | tail -1
  done
}
for S in FD001 FD002 FD003 FD004; do run_subset $S & done
wait
echo ALL_DONE
