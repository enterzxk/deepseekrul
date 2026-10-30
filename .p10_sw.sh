#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
chmod 644 data/CMAPSS/test_FD001.txt data/CMAPSS/RUL_FD001.txt
for ep in 12 20 30; do
  rm -rf experiments/p10/sweep/ep${ep}
  ( $PY -u scripts/run_ablation.py --subset FD001 --seeds 2027 --epochs $ep --patience 40 --window-size 50 --dropout 0.15 --weight-decay 1e-4 --asymmetric-weight 1.0 --lambda-mono 0.02 --late-weight 2.0 --clip-test-rul --overwrite --num-workers 2 --device cuda --variants p10_full --run-dir experiments/p10/sweep/ep${ep} > logs/p9/sweep10_${ep}.log 2>&1 ) &
done
wait
echo SWEEP10_DONE
