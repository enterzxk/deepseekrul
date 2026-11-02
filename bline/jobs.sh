#!/bin/bash
# B-line matrix driver.  Usage: jobs.sh <phase> <gpu>
#   phase = ind   in-distribution, 5 folds x 3 seeds x 7 variants
#   phase = lodo  leave-one-setpoint-out, 6 holds x 3 seeds x 7 variants
set -u
cd "$HOME/rulb"
PY=/home/user/miniconda3/envs/pmv/bin/python
export BL_WINDOW=40 BL_PARAMS=120000 BL_LR=1e-3 BL_PATIENCE=30
export CUDA_VISIBLE_DEVICES="${2:-0}"
export OMP_NUM_THREADS=4
mkdir -p experiments/bline/runs logs/bline
VARIANTS="erm erm_u condnorm cai cai_w dann coral"
phase="${1:-ind}"
if [ "$phase" = "ind" ]; then
  for v in $VARIANTS; do for f in 0 1 2 3 4; do for s in 2027 2028 2029; do
    echo "$v -1 $f $s"
  done; done; done
elif [ "$phase" = "ind2" ]; then
  for v in $VARIANTS; do for f in 0 1 2 3 4; do for s in 2027 2028; do
    echo "$v -1 $f $s"
  done; done; done
elif [ "$phase" = "lodo2" ]; then
  for h in 0 1 2 3 4 5; do for v in $VARIANTS; do for s in 2027 2028; do
    echo "$v $h 0 $s"
  done; done; done
elif [ "$phase" = "lodo_cai" ]; then
  for h in 0 1 2 3 4 5; do for v in cai cai_w; do for s in 2027 2028 2029; do
    echo "$v $h 0 $s"
  done; done; done
else
  for h in 0 1 2 3 4 5; do for v in $VARIANTS; do for s in 2027 2028 2029; do
    echo "$v $h 0 $s"
  done; done; done
fi
