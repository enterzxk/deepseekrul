#!/bin/bash
tail -3 ~/deepseek-rul/logs/mcd2_all.log
echo "--- per subset ---"
for S in FD001 FD002 FD003 FD004; do
  echo "[$S] $(tail -1 ~/deepseek-rul/logs/mcd2_$S.log 2>/dev/null | cut -c1-90)"
done
echo "--- files ---"
ls ~/deepseek-rul/experiments/punet_e2e/*/f0_s42/preds_mcd.csv 2>/dev/null | wc -l
