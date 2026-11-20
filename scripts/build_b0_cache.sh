#!/bin/bash
# Build B0 prediction cache (cuts + Delta scan) for LTAG v3 Phase 0.
set -u
cd "$HOME/sprint24" || exit 1
P=/home/user/miniconda3/envs/ncmapss/bin/python
export PYTHONPATH="$HOME/sprint24"
OUT="$HOME/sprint24/cache/preds"
LOGS="$HOME/sprint24/cache/logs"
mkdir -p "$LOGS"

build() {
  local subset="$1" seed="$2"
  local par="$OUT/B0/${subset}_s${seed}.parquet"
  if [[ -f "$par" ]]; then
    echo "SKIP $subset s$seed"
    return 0
  fi
  echo "=== $subset s$seed ==="
  "$P" -m paper_b.eval.build_pred_cache --subset "$subset" --seed "$seed" \
    --out "$OUT" > "$LOGS/${subset}_s${seed}.log" 2>&1
  tail -6 "$LOGS/${subset}_s${seed}.log"
}

for s in 101 202 303 404 505 606; do build FD001 "$s"; done
for subset in FD002 FD003 FD004; do
  for s in 101 202 303; do build "$subset" "$s"; done
done
echo "B0 CACHE DONE"
