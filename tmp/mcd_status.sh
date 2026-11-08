#!/bin/bash
echo "procs: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
echo "mcd files: $(ls ~/deepseek-rul/experiments/punet_e2e/*/f0_s42/mcdropout.json 2>/dev/null | wc -l)"
tail -2 ~/deepseek-rul/logs/mcd_all.log
for S in FD001 FD002 FD003 FD004; do
  echo "[$S] $(tail -1 ~/deepseek-rul/logs/mcd_$S.log 2>/dev/null | cut -c1-70)"
done
