#!/bin/bash
for S in FD001 FD002 FD003 FD004; do
  n=$(ps -eo cmd | grep -c "run_e2e_cqr.py --subset $S")
  echo "$S running=$n"
done
echo "--- point files ---"
ls ~/deepseek-rul/experiments/punet_e2e/*/f*_s42/e2e_point_none.json 2>/dev/null | wc -l
