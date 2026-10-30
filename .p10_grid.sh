#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
chmod 644 data/CMAPSS/test_FD001.txt data/CMAPSS/RUL_FD001.txt data/CMAPSS/test_FD002.txt data/CMAPSS/RUL_FD002.txt
V="tcn bilstm transformer p10_wo_crgcn p10_wo_synergy_mono p10_full"
$PY -u scripts/p8_run.py --subset FD001 --seeds 2027 2028 2029 2030 2031 --variants $V --epochs 40 --patience 20 --parallel 6 --full-train --out experiments/p10/final
echo FD001_P10_DONE
$PY -u scripts/p8_run.py --subset FD002 --seeds 2027 2028 2029 2030 2031 --variants $V --epochs 12 --patience 20 --parallel 6 --full-train --out experiments/p10/final
echo FD002_P10_DONE
