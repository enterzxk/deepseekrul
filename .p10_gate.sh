#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
chmod 644 data/CMAPSS/test_FD002.txt data/CMAPSS/RUL_FD002.txt
for ep in 10 12; do
  for v in p10_full transformer; do
    rm -rf experiments/p10/gate/${v}_ep${ep}
    ( $PY -u scripts/run_ablation.py --subset FD002 --seeds 2027 --epochs $ep --patience 20 --window-size 50 --dropout 0.15 --weight-decay 1e-4 --asymmetric-weight 1.0 --lambda-mono 0.02 --late-weight 2.0 --condition-aware --n-conditions 6 --clip-test-rul --overwrite --num-workers 2 --device cuda --variants $v --run-dir experiments/p10/gate/${v}_ep${ep} > logs/p9/gate10_${v}_${ep}.log 2>&1 ) &
  done
done
wait
echo GATE10_DONE
