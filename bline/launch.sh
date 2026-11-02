#!/bin/bash
# Usage: launch.sh <phase> <gpu> <parallel>
set -u
cd "$HOME/rulb"
export CUDA_VISIBLE_DEVICES="$2"
export BL_WINDOW=40 BL_PARAMS=120000 BL_LR=1e-3 BL_PATIENCE=30 OMP_NUM_THREADS=4
mkdir -p experiments/bline/runs logs/bline
bash bline/jobs.sh "$1" "$2" | xargs -P "${3:-4}" -n 4 sh -c \
  '/home/user/miniconda3/envs/pmv/bin/python bline/run.py --subset FD002 --variant $0 --hold $1 --fold $2 --seed $3 --epochs 200 > logs/bline/${0}_h${1}_f${2}_s${3}.log 2>&1'
echo "PHASE_${1}_DONE"
