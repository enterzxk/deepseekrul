#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
$PY scripts/run_punet_stage.py --subset FD001 --fold 0 --seed 42 --stage heads --reuse-ckpt 2>&1 | tail -8
