#!/bin/bash
echo "python procs: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
echo "bash workers: $(ps -eo cmd | grep -c '[r]un_matrix8')"
echo "--- log tail ---"
tail -5 ~/deepseek-rul/logs/matrix8.log 2>/dev/null | cut -c1-120
echo "--- counts ---"
echo "cqr $(ls ~/deepseek-rul/experiments/punet_matrix/*/f*_s*/preds_cqr.csv 2>/dev/null | wc -l) point $(ls ~/deepseek-rul/experiments/punet_matrix/*/f*_s*/preds_point.csv 2>/dev/null | wc -l)"
