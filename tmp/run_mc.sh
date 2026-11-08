#!/bin/bash
set -u
cd /home/user/rulb/tim_20261105/full_20261105
PY=/home/user/miniconda3/envs/pmv/bin/python
export OMP_NUM_THREADS=4
for cond in clean mc1 mc2; do
  $PY measurement_conditions.py FD002 2 M0 101 "$cond" --rundir runs3_w50_base 2>&1 | tail -1
done
