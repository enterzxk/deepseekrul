#!/usr/bin/env bash
cd "$HOME/deepseek-rul/experiments/ncmapss" || exit 1
echo '=== DS02: 10 most recently modified dirs ==='
ls -lat --time-style='+%Y-%m-%d %H:%M' DS02 | grep '^d' | head -10
echo
echo '=== mtimes of the cells the ledger says were appended ==='
for d in DS02_N0_LOEO-2_iv DS02_N0_LOEO-2_f200 DS02_N1_LOEO-2_s101_iv \
         DS02_N1_LOEO-2_s101_f200 DS02_N1_LOEO-2_s202_iv DS02_N1_LOEO-2_s202_f200; do
  printf '%-32s %s\n' "$d" "$(stat -c '%y' "DS02/$d/pred.csv" 2>&1 | cut -c1-19)"
done
echo
echo '=== DS06: 6 most recently modified dirs ==='
ls -lat --time-style='+%Y-%m-%d %H:%M' DS06 | grep '^d' | head -6
