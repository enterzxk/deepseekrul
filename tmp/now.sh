#!/bin/bash
cd ~/deepseek-rul
echo "cells: $(ls experiments/punet_matrix/*/f*_s*/preds_point.csv 2>/dev/null | wc -l)/40"
echo "cqr:   $(ls experiments/punet_matrix/*/f*_s*/preds_cqr.csv 2>/dev/null | wc -l)/40"
echo "mcd:   $(ls experiments/punet_matrix/*/f*_s*/preds_mcd.csv 2>/dev/null | wc -l)"
for S in FD001 FD002 FD003 FD004; do
  echo "  $S: $(ls experiments/punet_matrix/$S/f*_s*/preds_point.csv 2>/dev/null | wc -l) cells"
done
echo "procs: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
echo "seeds done: $(ls experiments/punet_matrix/*/f*_s7/preds_point.csv 2>/dev/null | wc -l) (seed 7)"
