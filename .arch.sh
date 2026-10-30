#!/bin/bash
set -e
cd /home/yndx/zxk/bearing-rul-pi || exit 1
OUT=/tmp/rul_archive; rm -rf $OUT; mkdir -p $OUT/predictions $OUT/weights
for root in order03 split8 split9; do
  for f in experiments/$root/*/FD002/*/matched_cutoff_predictions.csv; do
    [ -e "$f" ] || continue
    cell=$(echo "$f" | cut -d/ -f3)
    run=$(echo "$f" | cut -d/ -f5)
    cp "$f" "$OUT/predictions/${root}__${cell}__${run}__matched.csv"
  done
done
for root in order03 split8 split9; do
  for f in experiments/$root/*/FD002/*/best_model.pt; do
    [ -e "$f" ] || continue
    cell=$(echo "$f" | cut -d/ -f3)
    run=$(echo "$f" | cut -d/ -f5)
    cp "$f" "$OUT/weights/${root}__${cell}__${run}.pt"
  done
done
tar czf /tmp/rul_archive.tgz -C /tmp rul_archive
echo "predictions: $(ls $OUT/predictions | wc -l)  weights: $(ls $OUT/weights | wc -l)"
du -sh /tmp/rul_archive.tgz
