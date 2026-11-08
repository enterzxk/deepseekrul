#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
$PY scripts/run_e2e_cqr.py --subset FD001 --fold 0 --seed 42 --heads cqr --dump-preds   --prior-dir experiments/ida/prior_boot --out experiments/punet_e2e 2>&1 | tail -1
echo DUMP_DONE
