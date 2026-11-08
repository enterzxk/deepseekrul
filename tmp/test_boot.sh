#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
$PY scripts/build_prior_uncertainty.py --subsets FD001 --bootstrap 3 --out experiments/ida/prior_boot_test 2>&1 | tail -8
