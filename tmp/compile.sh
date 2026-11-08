#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
$PY -m py_compile scripts/run_femto_cqr.py scripts/evaluate_matrix.py scripts/run_e2e_cqr.py src/punet/conformal.py src/punet/evaluate.py && echo COMPILE_OK
echo "cqr dumps: $(ls experiments/punet_matrix/*/f*_s*/preds_cqr.csv 2>/dev/null | wc -l)/60"
