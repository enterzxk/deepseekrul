#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
nohup bash -c '
set -e
echo "=== retrain backbone with real sigma ==="
'"$PY"' scripts/run_punet_stage.py --subset FD001 --fold 0 --seed 42 --stage all --ablate none   --prior-dir experiments/ida/prior_boot --out experiments/punet_boot 2>&1 | tail -4
echo "=== A1 constant scale ==="
'"$PY"' scripts/run_punet_stage.py --subset FD001 --fold 0 --seed 42 --stage heads --reuse-ckpt --ablate A1   --prior-dir experiments/ida/prior_boot --out experiments/punet_boot 2>&1 | tail -2
echo "=== A4 permuted sigma ==="
'"$PY"' scripts/run_punet_stage.py --subset FD001 --fold 0 --seed 42 --stage heads --reuse-ckpt --ablate A4   --prior-dir experiments/ida/prior_boot --out experiments/punet_boot 2>&1 | tail -2
' > logs/validate_fd001.log 2>&1 &
echo "PID=$!"
sleep 10
tail -3 logs/validate_fd001.log 2>/dev/null || echo starting
