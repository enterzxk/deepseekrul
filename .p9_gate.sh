#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
chmod 644 data/CMAPSS/test_FD001.txt data/CMAPSS/RUL_FD001.txt data/CMAPSS/test_FD002.txt data/CMAPSS/RUL_FD002.txt
mkdir -p logs/p9
COMMON="--seeds 2027 --epochs 40 --patience 20 --window-size 50 --dropout 0.15 --weight-decay 1e-4 --asymmetric-weight 1.0 --lambda-mono 0.02 --late-weight 2.0 --clip-test-rul --overwrite --num-workers 2 --device cuda"
for ds in FD001 FD002; do
  EXTRA=""
  if [ "$ds" = "FD002" ]; then EXTRA="--condition-aware --n-conditions 6"; fi
  for v in p9_full transformer; do
    rm -rf experiments/p9/gate/${ds}_${v}
    ( $PY -u scripts/run_ablation.py --subset $ds $COMMON $EXTRA --variants $v --run-dir experiments/p9/gate/${ds}_${v} > logs/p9/gate_${ds}_${v}.log 2>&1 ) &
  done
done
wait
echo GATE91_DONE
