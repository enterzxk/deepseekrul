#!/bin/bash
set -u
cd /home/user/rulb/tim_20261105/full_20261105
deadline=$((SECONDS+21600))
while [ $SECONDS -lt $deadline ]; do
  missing=0
  for s in FD002 FD004; do for f in 0 1 2 3 4; do for a in M0 FULL; do
    [ -f runs3_w100_w100/$s/$f/${a}_101.csv ] || missing=1
  done; done; done
  [ $missing -eq 0 ] && break
  sleep 120
done
echo "wait ended $(date) missing=$missing" > logs/span_assemble.log
bash assemble_delivery.sh >> logs/span_assemble.log 2>&1
echo "SPAN ASSEMBLE DONE $(date)" >> logs/span_assemble.log
