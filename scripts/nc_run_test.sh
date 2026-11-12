#!/usr/bin/env bash
# Test phase: train on all six dev engines, score the sealed test engines.
#
# Sequential on purpose. One manifest covers all four subsets, and the gate
# records each open by read-modify-write of <freeze_id>.eval_start.json; two
# processes starting together would race on that file and one open would be lost.
# The batch is 40 cells with a single fit each, so the wall clock cost of not
# parallelising is minutes, not hours.
set -u
cd "$HOME/deepseek-rul" || exit 1
PY=/home/user/miniconda3/envs/ncmapss/bin/python
H5DIR=data/NCMAPSS/extracted
mkdir -p logs
for s in DS02 DS05 DS06 DS07; do
  h5=$(ls "$H5DIR"/N-CMAPSS_"$s"*.h5 | head -1)
  sha=$(sha256sum "$h5" | cut -d' ' -f1)
  echo "=== $s  $(date -Is)  $h5 ===" >> "logs/nc_test_${s}.log"
  "$PY" -m paper_b.ncmapss.run_loeo --h5 "$h5" --subset "$s" --phase test \
        --out-root experiments/ncmapss --data-sha256 "$sha" \
        >> "logs/nc_test_${s}.log" 2>&1
  echo "EXIT_${s}=$?" >> "logs/nc_test_${s}.log"
done
echo "ALL_DONE $(date -Is)"
