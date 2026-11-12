#!/bin/bash
# Stage D matrix COMMAND-LIST GENERATOR (a40 only). This script does NOT launch.
#
# The only executor for Stage D is 燕子's emitter (manager's ruling, 2026-11-13,
# after incident 5: one cell started by two executors). This script's whole job
# is to emit the 48 commands, with the admission guards attached, for that
# emitter to consume. The launching branch that used to live here has been
# removed, not disabled behind a flag -- a flag is one typo away from running.
set -u
cd /media/sangfor/vdb/user/zxk/deepseek-rul || exit 1
export PYTHONPATH=.
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH:-}
P=/home/yndx/miniconda3/envs/a40_env/bin/python
ROOT=experiments/hc/d
OUT=$ROOT/matrix_commands.txt
ARMS="B0 B1 B2 B3"
SUBSETS="FD001 FD002 FD003 FD004"
SEEDS="101 202 303"

# ---- admission guards (a failing guard means the list must not be used) ---- #
if [ ! -f "$ROOT/d45_init.json" ]; then
  echo "REFUSED: $ROOT/d45_init.json missing (c_T/c_S not frozen)" >&2; exit 2
fi
for s in $SUBSETS; do
  if [ ! -f "$ROOT/${s}_shared.json" ]; then
    echo "REFUSED: $ROOT/${s}_shared.json missing (run prepare)" >&2; exit 2
  fi
done

$P - "$ROOT" <<'PY' || exit 3
import json, sys
fz = json.load(open(sys.argv[1] + '/d45_init.json'))['frozen']
print('# frozen  c_T=%.6f c_S=%.6f  lam_T=%.4f lam_S=%.4f'
      % (fz['c_T'], fz['c_S'], fz['lam_T'], fz['lam_S']))
PY

: > "$OUT"
n=0
for s in $SUBSETS; do
  for a in $ARMS; do
    for sd in $SEEDS; do
      n=$((n+1))
      # the run id comes from the same function the trainer uses for the output
      # directory, so a registered id can never point at a different directory
      rid=$($P -m paper_b.hc.train_d id --arm "$a" --subset "$s" --seed "$sd") || exit 4
      # `-u` is not cosmetic: stdout redirected to a file is block-buffered at
      # 4 KB, the trainer prints ~60 B per epoch and a cell runs at most 40
      # epochs, so without -u the log stays EMPTY for the whole 6-10 h and a
      # hung run is indistinguishable from a healthy one. It also means a
      # mid-run crash leaves no traceback anywhere.
      echo "$P -u -m paper_b.hc.train_d train --subset $s --arm $a --seed $sd --root $ROOT"
      echo "#   run_id=$rid  dir=$ROOT/$rid" >&2
    done
  done
done >> "$OUT"
echo "cells: $n  -> $OUT" >&2
echo "NOT LAUNCHED. The only executor is the group's single emitter." >&2
