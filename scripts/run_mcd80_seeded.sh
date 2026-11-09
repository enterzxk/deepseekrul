#!/bin/bash
# Seeded re-pass of MC-Dropout over all 80 cells.
#
# Why: the first pass was unseeded, so its intervals are not reproducible
# (rerun on identical checkpoints moved picp by up to 7.6e-3 and Winkler by
# up to 3.0). This pass fixes RNG = cell seed and writes to a separate root so
# the first pass is preserved. The difference between the two passes, over all
# 80 cells, is the Monte-Carlo noise floor for this baseline.
set -u
cd ~/deepseek-rul || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH:-}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
OUT=experiments/punet_mcd80_seeded
MAT=experiments/punet_matrix
FIRST=experiments/punet_mcd80
mkdir -p "$OUT" logs
LOG=logs/mcd80_seeded.log
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

run_one() { # subset fold seed ckpt_root
  S=$1; F=$2; SD=$3; R=$4
  local D="$OUT/$S/f${F}_s${SD}"
  mkdir -p "$D"
  [ -f "$D/mcdropout.json" ] && return 0
  $PY scripts/run_mcdropout.py --subset "$S" --fold "$F" --seed "$SD" \
      --prior-dir experiments/ida/prior_boot --root "$R" --out "$OUT" \
      >> "logs/mcdS_${S}_f${F}_s${SD}.log" 2>&1
  say "seeded mcd $S f$F s$SD root=$R rc=$?"
}

say "SEEDED PASS start"
for S in FD001 FD002 FD003 FD004; do
  (
    for SD in 42 7; do for F in 0 1 2 3 4; do run_one "$S" "$F" "$SD" "$MAT"; done; done
    for SD in 2024 13; do for F in 0 1 2 3 4; do run_one "$S" "$F" "$SD" "$FIRST"; done; done
    say "seeded worker $S done"
  ) &
done
wait
say "SEEDED PASS done; cells=$(ls $OUT/*/f*_s*/mcdropout.json 2>/dev/null | wc -l)"
