#!/bin/bash
cd ~/deepseek-rul
echo "procs: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
for S in FD001 FD002 FD003 FD004; do
  echo "  $S: s2024=$(ls experiments/punet_matrix/$S/f*_s2024/preds_point.csv 2>/dev/null | wc -l)/5  s13=$(ls experiments/punet_matrix/$S/f*_s13/preds_point.csv 2>/dev/null | wc -l)/5"
done
echo "total cells: $(ls experiments/punet_matrix/*/f*_s*/preds_point.csv 2>/dev/null | wc -l)"
tail -2 logs/seeds_more.log
