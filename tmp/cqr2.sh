#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for S in FD001 FD004; do
  R=$($PY scripts/run_punet_stage.py --subset $S --fold 0 --seed 42 --stage heads --reuse-ckpt     --prior-dir experiments/ida/prior_boot --out experiments/punet_5fold --method cqr     --head-lr 1e-3 --head-epochs 150 2>&1 | tail -1)
  echo "$S lr1e-3 $R"
done
