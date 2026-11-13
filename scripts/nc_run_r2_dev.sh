#!/usr/bin/env bash
# Round-2 dev grid: 78 cells per subset x 2 evaluation points = 156 rows/subset.
#
# Output goes to experiments/ncmapss_r2, NOT experiments/ncmapss: round 1's
# directory names are derived from (subset, rung, fold, seed) and would collide,
# which is how the first attempt overwrote six round-1 cells per subset. The
# ledger is passed explicitly for the same reason.
set -u
cd "$HOME/deepseek-rul" || exit 1
PY=/home/user/miniconda3/envs/ncmapss/bin/python
H5DIR=data/NCMAPSS/extracted
ROOT=experiments/ncmapss_r2
mkdir -p logs

run_pair() {
  local gpu="$1"; shift
  for s in "$@"; do
    local h5 sha
    h5=$(ls "$H5DIR"/N-CMAPSS_"$s"*.h5 | head -1)
    sha=$(sha256sum "$h5" | cut -d' ' -f1)
    echo "=== $s on GPU $gpu  $(date -Is)  $h5 ===" >> "logs/nc_r2_${s}.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" -m paper_b.ncmapss.run_loeo \
      --h5 "$h5" --subset "$s" --out-root "$ROOT" \
      --ledger "$ROOT/$s/runs_${s}_r2.csv" \
      --data-sha256 "$sha" >> "logs/nc_r2_${s}.log" 2>&1
    echo "EXIT_${s}=$?" >> "logs/nc_r2_${s}.log"
  done
}

run_pair 0 DS02 DS05 &
run_pair 1 DS06 DS07 &
wait
echo "ALL_DONE $(date -Is)"
