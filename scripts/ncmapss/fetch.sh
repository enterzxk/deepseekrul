#!/bin/bash
# Resumable fetch of the NASA N-CMAPSS archive (15.76 GB, 8 parallel ranges).
#
# The archive is one deflate stream containing a single 15.76 GB member, so no byte range
# of the inner zip can be read without decompressing everything before it. Splitting the
# OUTER file into ranges is still worth it because a single connection tops out around
# 5 MB/s here while eight reach ~40 MB/min aggregate on a good hour.
#
# Every part resumes with -C -, so re-running after an interruption costs nothing. Sizes are
# checked before the parts are concatenated: a short part silently corrupts the concatenation
# and the resulting zip fails only at the very end, after another hour of work.
set -u
URL="https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip"
TOTAL=15760443389
N=8
DIR="${1:-/tmp/ncmapss}"
mkdir -p "$DIR"
CH=$(( (TOTAL + N - 1) / N ))

for i in $(seq 0 $((N-1))); do
  S=$((i*CH)); E=$((S+CH-1)); [ "$E" -ge "$TOTAL" ] && E=$((TOTAL-1))
  WANT=$((E-S+1))
  HAVE=0
  [ -f "$DIR/part$i.bin" ] && HAVE=$(stat -c %s "$DIR/part$i.bin")
  if [ "$HAVE" -ge "$WANT" ]; then
    echo "part$i complete ($HAVE/$WANT)"
    continue
  fi
  (
    curl -sS --retry 5 --retry-delay 3 -C - -r "$S-$E" -o "$DIR/part$i.bin" "$URL" \
      && echo "part$i done"
  ) &
done
wait

# verify before concatenating
ok=1
for i in $(seq 0 $((N-1))); do
  S=$((i*CH)); E=$((S+CH-1)); [ "$E" -ge "$TOTAL" ] && E=$((TOTAL-1))
  WANT=$((E-S+1)); HAVE=$(stat -c %s "$DIR/part$i.bin" 2>/dev/null || echo 0)
  if [ "$HAVE" -ne "$WANT" ]; then echo "part$i SHORT: $HAVE != $WANT"; ok=0; fi
done
[ "$ok" -eq 1 ] || { echo "refusing to concatenate"; exit 1; }

cat "$DIR"/part0.bin "$DIR"/part1.bin "$DIR"/part2.bin "$DIR"/part3.bin \
    "$DIR"/part4.bin "$DIR"/part5.bin "$DIR"/part6.bin "$DIR"/part7.bin > "$DIR/full.bin"
SZ=$(stat -c %s "$DIR/full.bin")
echo "full.bin = $SZ bytes (expected $TOTAL)"
[ "$SZ" -eq "$TOTAL" ] || exit 1
