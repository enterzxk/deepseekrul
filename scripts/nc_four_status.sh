#!/usr/bin/env bash
# Progress of the four-subset dev grid: rows, checkpoints, and both rules per subset.
set -u
cd "$HOME/deepseek-rul" || exit 1
echo "HEAD=$(git rev-parse --short HEAD)  procs=$(pgrep -fc 'paper_b.ncmapss.run_loeo --h5' || true)"
for s in DS02 DS05 DS06 DS07; do
  f="experiments/ncmapss/$s/runs_dev.csv"
  if [ -f "$f" ]; then
    rows=$(($(wc -l < "$f") - 1))
    ck=$(find "experiments/ncmapss/$s" -name 'ckpt_*.pt' 2>&1 | wc -l)
    iv=$(find "experiments/ncmapss/$s" -maxdepth 1 -type d -name '*_iv' 2>&1 | wc -l)
    f2=$(find "experiments/ncmapss/$s" -maxdepth 1 -type d -name '*_f200' 2>&1 | wc -l)
    echo "$s rows=$rows ckpt=$ck dirs_iv=$iv dirs_f200=$f2"
  else
    echo "$s: no ledger yet"
  fi
done
