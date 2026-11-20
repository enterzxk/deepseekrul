#!/bin/bash
set -u
cd "$HOME/sprint24" || exit 1
P=/home/user/miniconda3/envs/ncmapss/bin/python
export PYTHONPATH="$HOME/sprint24"
for s in FD001 FD002 FD003 FD004; do
  echo "=== $s ==="
  "$P" -m paper_b.eval.ltagv2_phase3 --subset "$s" --out "results/ltagv2_p3_${s}.json" 2>&1 | tail -2
done
echo "P3 DONE"
