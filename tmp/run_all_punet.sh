#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
nohup bash -c '
for S in FD001 FD002 FD003 FD004; do
  echo "=== $S backbone ==="
  '"$PY"' scripts/run_punet_stage.py --subset $S --fold 0 --seed 42 --stage all --ablate none     --prior-dir experiments/ida/prior_boot --out experiments/punet_boot 2>&1 | tail -2
  for A in A1 A4; do
    echo "=== $S $A ==="
    '"$PY"' scripts/run_punet_stage.py --subset $S --fold 0 --seed 42 --stage heads --reuse-ckpt       --ablate $A --prior-dir experiments/ida/prior_boot --out experiments/punet_boot 2>&1 | tail -1
  done
done' > logs/punet_all.log 2>&1 &
echo "PID=$!"
sleep 6
tail -2 logs/punet_all.log 2>/dev/null || echo starting
