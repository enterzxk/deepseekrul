#!/usr/bin/env bash
# Restore the DS06 cells whose round-1 directories were overwritten, from the
# round-1 rebuild already produced under ad53371 in /tmp/r1out.
#
# Acceptance is metric AND ckpt_sha256 against the sealed row (verified separately
# by scripts/nc_verify_restore.py: 8/8 both). The ledger loses the 6 rows the
# accidental round-2 start appended.
set -eu
cd "$HOME/deepseek-rul" || exit 1
DST=experiments/ncmapss/DS06
SRC=/tmp/r1out/DS06
for d in N0_LOEO-1_iv N0_LOEO-1_f200 N1_LOEO-1_s101_iv N1_LOEO-1_s101_f200 \
         N1_LOEO-1_s202_iv N1_LOEO-1_s202_f200 N1_LOEO-1_s303_iv N1_LOEO-1_s303_f200; do
  [ -d "$SRC/DS06_$d" ] || { echo "missing $SRC/DS06_$d"; exit 1; }
  rm -rf "$DST/DS06_$d"
  cp -a "$SRC/DS06_$d" "$DST/"
done
echo "copied 8 dirs into $DST"
# the round-2 rows were appended at the end; round 1 had 120 data rows
rows=$(($(wc -l < "$DST/runs_dev.csv") - 1))
if [ "$rows" -gt 120 ]; then
  head -121 "$DST/runs_dev.csv" > /tmp/ds06_trim.csv
  mv /tmp/ds06_trim.csv "$DST/runs_dev.csv"
  echo "trimmed ledger $rows -> 120 rows"
fi
echo "final rows=$(($(wc -l < "$DST/runs_dev.csv") - 1))"
