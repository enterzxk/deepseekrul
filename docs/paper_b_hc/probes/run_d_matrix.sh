#!/bin/bash
# Stage D four-arm matrix dispatcher (a40 only).
#
# Admission is NOT implied by this script: it refuses to run unless the frozen
# calibration record exists and every shared record carries the frozen weights.
# Use --dry-run to print the exact command list without starting anything.
#
# Cells: 4 arms x 4 subsets x 3 seeds = 48. Every cell reads the same
# <subset>_shared.json (l_mean, b0, S0, head fingerprint) and the same frozen
# c_T/c_S, so the only per-run variable is the training seed.
set -u
cd /media/sangfor/vdb/user/zxk/deepseek-rul || exit 1
export PYTHONPATH=.
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH:-}
P=/home/yndx/miniconda3/envs/a40_env/bin/python
ROOT=experiments/hc/d
LOG=$ROOT/matrix.log
LEDGER=$ROOT/ledger.csv
ARMS="B0 B1 B2 B3"
SUBSETS="FD001 FD002 FD003 FD004"
SEEDS="101 202 303"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# ---- admission guards ----------------------------------------------------- #
if [ ! -f "$ROOT/d45_init.json" ]; then
  echo "REFUSED: $ROOT/d45_init.json missing (c_T/c_S not frozen)"; exit 2
fi
for s in $SUBSETS; do
  if [ ! -f "$ROOT/${s}_shared.json" ]; then
    echo "REFUSED: $ROOT/${s}_shared.json missing (run prepare)"; exit 2
  fi
done
$P - "$ROOT" <<'PY' || exit 3
import json, sys
root = sys.argv[1]
fz = json.load(open(root + '/d45_init.json'))['frozen']
print('frozen  c_T=%.6f c_S=%.6f  lam_T=%.4f lam_S=%.4f'
      % (fz['c_T'], fz['c_S'], fz['lam_T'], fz['lam_S']))
PY

n=0
for s in $SUBSETS; do for a in $ARMS; do for sd in $SEEDS; do n=$((n+1)); done; done; done
echo "cells: $n  (arms x subsets x seeds)"
if [ $DRY -eq 0 ] && [ ! -f "$LEDGER" ]; then
  echo "arm,subset,seed,K,best_ep,val_engine_mean_rmse,val_pooled_rmse,rmse_uncensored,lam_source,head_fingerprint" > "$LEDGER"
fi
[ $DRY -eq 1 ] && { echo "--- dry run, nothing started ---"; }

cell_done() {
  # A cell counts as done only if it is a FORMAL run that closed normally:
  # no budget_override, frozen weights, right K. A smoke run that happens to
  # have written metrics.json must never be mistaken for the formal cell.
  local mr="$1/metrics.json"
  [ -f "$mr" ] || return 1
  $P - "$mr" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
ok = (m.get('budget_override') is None
      and 'frozen' in str(m.get('lam_source'))
      and int(m.get('K', -1)) == 600
      and m.get('frozen_sha256'))
sys.exit(0 if ok else 1)
PY
}

run_cell() {
  local arm=$1 subset=$2 seed=$3
  local out="$ROOT/${arm}_${subset}_s${seed}"
  if cell_done "$out"; then
    echo "SKIP $arm $subset s$seed (formal cell already closed)" | tee -a "$LOG"
    return 0
  fi
  if [ -f "$out/metrics.json" ]; then
    echo "STALE $arm $subset s$seed (metrics.json is not a formal closed run; move it aside)" | tee -a "$LOG"
    return 4
  fi
  local cmd="$P -m paper_b.hc.train_d train --subset $subset --arm $arm --seed $seed --root $ROOT"
  if [ $DRY -eq 1 ]; then echo "$cmd"; return 0; fi
  echo "=== $(date -u +%FT%TZ) $arm $subset s$seed ===" | tee -a "$LOG"
  $cmd >> "$LOG" 2>&1
  local rc=$?
  local mr="$out/metrics.json"
  if [ $rc -eq 0 ] && [ -f "$mr" ]; then
    $P - "$mr" "$arm" "$subset" "$seed" >> "$LEDGER" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' % (
    sys.argv[2], sys.argv[3], sys.argv[4], m['K'], m['best_ep'],
    '%.4f' % m['val_engine_mean_rmse'], '%.4f' % m['val_pooled_rmse'],
    '%.6f' % m['rmse_uncensored'] if m['rmse_uncensored'] is not None else '',
    m['lam_source'], m['head_fingerprint']))
PY
  else
    echo "FAIL $arm $subset s$seed rc=$rc" | tee -a "$LOG"
  fi
}

for s in $SUBSETS; do
  for a in $ARMS; do
    for sd in $SEEDS; do
      run_cell "$a" "$s" "$sd"
    done
  done
done
echo "DONE" | tee -a "$LOG"
