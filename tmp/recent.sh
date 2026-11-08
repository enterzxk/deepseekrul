#!/bin/bash
cd ~/deepseek-rul/experiments/punet_matrix
echo "--- last 8 completions ---"
ls -lt --time-style=+%H:%M */*/e2e_*_none.json 2>/dev/null | head -8 | awk '{print $6, $7}'
echo "--- now ---"; date +%H:%M
echo "--- running ---"
ps -eo etime,cmd | grep '[r]un_e2e_cqr' | cut -c1-60
echo "--- counts ---"
echo "cqr $(ls */*/preds_cqr.csv 2>/dev/null | wc -l) point $(ls */*/preds_point.csv 2>/dev/null | wc -l)"
