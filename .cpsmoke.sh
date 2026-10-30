#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
$PY - <<'EOF'
import ast; ast.parse(open("scripts/run_candidate.py", encoding="utf-8").read()); print("syntax ok")
EOF
$PY -u scripts/run_candidate.py --subset FD002 --variant A --seed 2027 --split-seed 7 \
  --ref-mode global --cp-labels --epochs 1 --patience 1 --d-model 64 --num-layers 1 \
  --out experiments/cpcheck 2>&1 | tail -n 3
