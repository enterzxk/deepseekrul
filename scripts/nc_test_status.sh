#!/usr/bin/env bash
# Progress of the test-phase batch: rows, checkpoints, and the last log lines.
set -u
cd "$HOME/deepseek-rul" || exit 1
echo "procs=$(pgrep -fc 'paper_b.ncmapss.run_loeo --h5' || true)"
for s in DS02 DS05 DS06 DS07; do
  f="experiments/ncmapss/$s/runs_test.csv"
  if [ -f "$f" ]; then
    rows=$(($(wc -l < "$f") - 1))
    ck=$(find "experiments/ncmapss/$s" -maxdepth 2 -name 'ckpt_*.pt' 2>/dev/null | wc -l)
    echo "$s rows=$rows (expected 20)  ckpt_total=$ck"
  else
    echo "$s: not started"
  fi
done
echo '--- last lines per subset ---'
for s in DS02 DS05 DS06 DS07; do
  [ -f "logs/nc_test_${s}.log" ] || continue
  echo "[$s] $(tail -1 "logs/nc_test_${s}.log")"
done
