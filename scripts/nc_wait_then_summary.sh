#!/usr/bin/env bash
# Wait for the full dev run to finish, then write the paired two-column summary.
set -u
cd "$HOME/deepseek-rul" || exit 1
while pgrep -f 'paper_b.ncmapss.run_loeo --h5' > /dev/null; do
  sleep 60
done
echo "run finished at $(date -Is)"
python3 scripts/nc_two_col_summary.py \
        --ledger experiments/ncmapss/runs_dev.csv \
        --out-root experiments/ncmapss > logs/nc_two_col_summary.txt 2>&1
echo "SUMMARY_EXIT=$?" >> logs/nc_two_col_summary.txt
