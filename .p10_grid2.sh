#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
V="tcn bilstm transformer p10_wo_crgcn p10_wo_synergy_mono p10_full"
$PY -u scripts/p8_run.py --subset FD001 --seeds 2027 2028 2029 2030 2031 --variants $V --epochs 30 --patience 30 --parallel 6 --full-train --out experiments/p10/final2
echo FD001_FINAL2_DONE
$PY -u scripts/p8_run.py --subset FD002 --seeds 2027 2028 2029 2030 2031 --variants $V --epochs 5 --patience 5 --parallel 6 --full-train --out experiments/p10/final2
echo FD002_FINAL2_DONE
