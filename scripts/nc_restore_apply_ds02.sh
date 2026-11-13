#!/usr/bin/env bash
# Same restore for DS02's overwritten cells, and trim its ledger back to 120 rows.
# The DS02 sealed table is the pilot batch (_ev naming, no subset prefix), so these
# four-subset rows are not verifiable against it; the rebuild is still the same
# deterministic code path, and it returns the directory to its round-1 state.
set -eu
cd "$HOME/deepseek-rul" || exit 1
DST=experiments/ncmapss/DS02
SRC=/tmp/r1out/DS02
for d in N0_LOEO-2_iv N0_LOEO-2_f200 N1_LOEO-2_s101_iv N1_LOEO-2_s101_f200 \
         N1_LOEO-2_s202_iv N1_LOEO-2_s202_f200 N1_LOEO-2_s303_iv N1_LOEO-2_s303_f200; do
  [ -d "$SRC/DS02_$d" ] || { echo "missing $SRC/DS02_$d"; exit 1; }
  rm -rf "$DST/DS02_$d"
  cp -a "$SRC/DS02_$d" "$DST/"
done
echo "copied 8 dirs into $DST"
rows=$(($(wc -l < "$DST/runs_dev.csv") - 1))
if [ "$rows" -gt 120 ]; then
  head -121 "$DST/runs_dev.csv" > /tmp/ds02_trim.csv
  mv /tmp/ds02_trim.csv "$DST/runs_dev.csv"
  echo "trimmed ledger $rows -> 120 rows"
fi
echo "final rows=$(($(wc -l < "$DST/runs_dev.csv") - 1))"
