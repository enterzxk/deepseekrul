#!/bin/bash
# MC-Dropout 80-cell backfill (D009 step 3 / D016 evidence base).
#
# Pre-registered gate (written BEFORE launch, see docs/decision_log.md D019):
#   H-MCD80: dropout-sampled predictive std (T=30) is an effective conditional
#            scale for conformal RUL intervals.
#   Metric : paired Winkler at coverage-aligned 0.90, alpha FIXED at 0.1
#            (D005 protocol), MC-Dropout vs constant width, per cell.
#   Unit   : fold (n=20), seeds nested; 80 cells are NOT independent.
#   PASS   : mcd wins >=60% of folds AND mean rel. delta <= -5% AND
#            fold-level bootstrap 95% CI excludes 0 -> retract "not effective".
#   FAIL   : mcd loses >=70% of folds -> scope-limited counterexample holds.
#   ELSE   : subset-conditional statement only, no global claim.
#   INTEGRITY: if a retrained point model does not reproduce the frozen
#            preds_point.csv (max |dy_lo| > 1e-3) the paired comparison is
#            invalid -> stop, escalate to L2, do not mix.
#
# Budget: <=6 GPU-hours on the single A40; 40 point retrains + 80 MC-Dropout.
# Resumable: every step skips when its output already exists.
set -u
cd ~/deepseek-rul || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH:-}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
OUT=experiments/punet_mcd80
MAT=experiments/punet_matrix
mkdir -p "$OUT" logs
LOG=logs/mcd80.log
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

mcd_one() { # subset fold seed ckpt_root
  S=$1; F=$2; SD=$3; R=$4
  local D="$OUT/$S/f${F}_s${SD}"
  mkdir -p "$D"
  if [ -f "$D/mcdropout.json" ]; then say "skip mcd $S f$F s$SD"; return 0; fi
  $PY scripts/run_mcdropout.py --subset "$S" --fold "$F" --seed "$SD" \
      --prior-dir experiments/ida/prior_boot --root "$R" --out "$OUT" \
      >> "logs/mcd_${S}_f${F}_s${SD}.log" 2>&1
  local rc=$?
  say "mcd $S f$F s$SD root=$R rc=$rc"
  return $rc
}

train_point() { # subset fold seed
  S=$1; F=$2; SD=$3
  local D="$OUT/$S/f${F}_s${SD}"
  mkdir -p "$D"
  if [ -f "$D/model.pt" ]; then say "skip train $S f$F s$SD"; return 0; fi
  $PY scripts/run_e2e_cqr.py --subset "$S" --fold "$F" --seed "$SD" --heads point \
      --dump-preds --save-ckpt --prior-dir experiments/ida/prior_boot --out "$OUT" \
      >> "logs/train_${S}_f${F}_s${SD}.log" 2>&1
  local rc=$?
  say "train $S f$F s$SD rc=$rc"
  return $rc
}

# ---- Stage A: MC-Dropout on the 40 cells that already have checkpoints ----
say "STAGE A start (seeds 42,7; ckpt root=$MAT)"
for S in FD001 FD002 FD003 FD004; do
  (
    for SD in 42 7; do for F in 0 1 2 3 4; do mcd_one "$S" "$F" "$SD" "$MAT"; done; done
    say "STAGE A worker $S done"
  ) &
done
wait
say "STAGE A done; cells=$(ls $OUT/*/f*_s*/mcdropout.json 2>/dev/null | wc -l)"

# ---- Stage B: retrain the 40 missing point models, then MC-Dropout ----
say "STAGE B start (seeds 2024,13; retrain point + save ckpt)"
for S in FD001 FD002 FD003 FD004; do
  (
    for SD in 2024 13; do
      for F in 0 1 2 3 4; do
        train_point "$S" "$F" "$SD" && mcd_one "$S" "$F" "$SD" "$OUT"
      done
    done
    say "STAGE B worker $S done"
  ) &
done
wait
say "STAGE B done; cells=$(ls $OUT/*/f*_s*/mcdropout.json 2>/dev/null | wc -l)"

# ---- Stage C: integrity verification against the frozen baseline ----
$PY scripts/verify_mcd80.py >> "$LOG" 2>&1
say "STAGE C verify rc=$?"
say "ALL_DONE cells=$(ls $OUT/*/f*_s*/mcdropout.json 2>/dev/null | wc -l)"
