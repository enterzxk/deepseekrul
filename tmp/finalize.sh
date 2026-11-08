#!/bin/bash
set -u
cd /home/user/rulb/tim_20261105/full_20261105
deadline=$((SECONDS+21600))
while [ $SECONDS -lt $deadline ]; do
  missing=0
  for s in 202 303; do for f in 0 1 2 3 4; do for a in M0 FULL; do
    [ -f runs/FD002/$f/${a}_$s.csv ] || missing=1
  done; done; done
  [ -f runs/FD004/4/FULL_101.csv ] || missing=1
  [ $missing -eq 0 ] && break
  sleep 120
done
echo "wait loop ended $(date) missing=$missing" > logs/assemble.log
bash assemble_delivery.sh >> logs/assemble.log 2>&1
echo "FINALIZE DONE $(date)" >> logs/assemble.log
