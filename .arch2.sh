#!/bin/bash
set -e
cd /home/yndx/zxk/bearing-rul-pi || exit 1
OUT=/tmp/rul_archive2; rm -rf $OUT; mkdir -p $OUT/predictions $OUT/weights
for f in experiments/*/*/FD002/*/matched_cutoff_predictions.csv; do
  [ -e "$f" ] || continue
  root=$(echo "$f" | cut -d/ -f2); cell=$(echo "$f" | cut -d/ -f3); run=$(echo "$f" | cut -d/ -f5)
  cp "$f" "$OUT/predictions/${root}__${cell}__${run}__matched.csv"
done
for f in experiments/*/*/FD002/*/best_model.pt; do
  [ -e "$f" ] || continue
  root=$(echo "$f" | cut -d/ -f2); cell=$(echo "$f" | cut -d/ -f3); run=$(echo "$f" | cut -d/ -f5)
  cp "$f" "$OUT/weights/${root}__${cell}__${run}.pt"
done
cd /tmp && tar czf /tmp/rul_archive2.tgz rul_archive2
echo "predictions=$(ls $OUT/predictions | wc -l) weights=$(ls $OUT/weights | wc -l)"
