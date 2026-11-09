#!/bin/bash
# 80-cell frozen matrix launcher (4 subsets x 5 folds x 4 seeds x 2 heads).
#
# PROVENANCE: reconstructed verbatim on 2026-11-09 from the untracked scratch
# scripts tmp/matrix2seeds.sh (Stage A: seeds 42, 7) and tmp/run_more_seeds.sh
# (Stage B: seeds 2024, 13).  Those files lived under /tmp/, which .gitignore
# excludes, so the exact command lines that produced the 80 cells were not
# under version control.  See docs/BASELINE_FREEZE.md sections 5.2 (B-1) and 7.5.
#
# This script only *re-creates missing cells*; it never overwrites an existing
# preds_*.csv.  Run it on a40 (no network, no git):
#
#   ssh a40
#   cd ~/deepseek-rul
#   export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
#   bash scripts/launch_matrix_80cells.sh            # both stages
#   bash scripts/launch_matrix_80cells.sh A          # seeds 42, 7 only
#   bash scripts/launch_matrix_80cells.sh B          # seeds 2024, 13 only
#
# Budget: ~20 min per cell on a40 -> ~27 GPU-h for all 80 cells.
# Gate: after completion, `python scripts/freeze_audit.py` must report 50/50
# (B1-B6 reproduce the frozen 75/80, -9.087%, -10.754%).
set -u
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-/home/yndx/miniconda3/envs/a40_env/bin/python}
PRIOR=${PRIOR:-experiments/ida/prior_boot}
OUT=${OUT:-experiments/punet_matrix}
STAGE=${1:-ALL}

run_cell() {  # $1=subset $2=fold $3=seed $4=heads $5=extra-flags
  local S=$1 F=$2 SEED=$3 HEADS=$4 EXTRA=$5
  local D="$OUT/$S/f${F}_s${SEED}"
  local P="$D/preds_${HEADS}.csv"
  [ -f "$P" ] && return 0
  mkdir -p "$D"
  echo "=== $S f$F s$SEED $HEADS ==="
  # shellcheck disable=SC2086
  "$PY" scripts/run_e2e_cqr.py --subset "$S" --fold "$F" --seed "$SEED" \
      --heads "$HEADS" --dump-preds $EXTRA \
      --prior-dir "$PRIOR" --out "$OUT" 2>&1 | tail -1
}

worker() {
  local S=$1
  if [ "$STAGE" = "ALL" ] || [ "$STAGE" = "A" ]; then
    for SEED in 42 7; do for F in 0 1 2 3 4; do
      run_cell "$S" "$F" "$SEED" cqr   "--save-ckpt"
      run_cell "$S" "$F" "$SEED" point "--save-ckpt"
    done; done
  fi
  if [ "$STAGE" = "ALL" ] || [ "$STAGE" = "B" ]; then
    for SEED in 2024 13; do for F in 0 1 2 3 4; do
      run_cell "$S" "$F" "$SEED" cqr   ""
      run_cell "$S" "$F" "$SEED" point ""
    done; done
  fi
  echo "DONE $S"
}

for S in FD001 FD002 FD003 FD004; do worker "$S" & done
wait
echo ALL_DONE
