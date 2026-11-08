#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
echo "cqr $(ls experiments/punet_matrix/*/f*_s*/preds_cqr.csv 2>/dev/null | wc -l) point $(ls experiments/punet_matrix/*/f*_s*/preds_point.csv 2>/dev/null | wc -l)"
/home/yndx/miniconda3/envs/a40_env/bin/python scripts/evaluate_matrix.py 2>&1 | tail -12
