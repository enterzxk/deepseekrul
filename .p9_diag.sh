#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
COMMON="--subset FD002 --seeds 2027 --epochs 40 --patience 20 --window-size 50 --dropout 0.15 --weight-decay 1e-4 --asymmetric-weight 1.0 --lambda-mono 0.02 --late-weight 2.0 --condition-aware --n-conditions 6 --clip-test-rul --overwrite --num-workers 2 --device cuda"
for v in p9_wo_crgcn p9_wo_synergy_mono; do
  rm -rf experiments/p9/gate/FD002_${v}
  ( $PY -u scripts/run_ablation.py $COMMON --variants $v --run-dir experiments/p9/gate/FD002_${v} > logs/p9/gate_FD002_${v}.log 2>&1 ) &
done
wait
echo DIAG9_DONE
