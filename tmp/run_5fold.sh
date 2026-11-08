#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
nohup bash -c '
for S in FD001 FD002 FD003 FD004; do
  for F in 0 1 2 3 4; do
    echo "=== $S f$F backbone ==="
    '"$PY"' scripts/run_punet_stage.py --subset $S --fold $F --seed 42 --stage all --ablate none       --prior-dir experiments/ida/prior_boot --out experiments/punet_5fold 2>&1 | tail -1
    for A in A1 A4; do
      echo "=== $S f$F $A ==="
      '"$PY"' scripts/run_punet_stage.py --subset $S --fold $F --seed 42 --stage heads --reuse-ckpt         --ablate $A --prior-dir experiments/ida/prior_boot --out experiments/punet_5fold 2>&1 | tail -1
    done
  done
done' > logs/punet_5fold.log 2>&1 &
echo "PID=$!"
sleep 5
tail -2 logs/punet_5fold.log 2>/dev/null || echo starting
