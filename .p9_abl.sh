#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for ds in FD001 FD002; do
  EXTRA=""; [ "$ds" = "FD002" ] && EXTRA="--condition-aware --n-conditions 6"
  rm -rf experiments/p9/gate/${ds}_p9_wo_synergy_mono experiments/p9/gate/${ds}_p9_wo_crgcn
  ( $PY -u scripts/run_ablation.py --subset $ds --seeds 2027 --epochs 40 --patience 20 --window-size 50 --dropout 0.15 --weight-decay 1e-4 --asymmetric-weight 1.0 --lambda-mono 0.02 --late-weight 2.0 --clip-test-rul --overwrite --num-workers 2 --device cuda $EXTRA --variants p9_wo_synergy_mono p9_wo_crgcn --run-dir experiments/p9/gate/${ds}_abl > logs/p9/abl_${ds}.log 2>&1 ) &
done
wait
echo ABL9_DONE
