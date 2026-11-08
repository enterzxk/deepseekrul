#!/bin/bash
echo "procs: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
echo "cqr dumps:  $(ls ~/deepseek-rul/experiments/punet_matrix/*/f*_s*/preds_cqr.csv 2>/dev/null | wc -l)/60"
echo "point dumps:$(ls ~/deepseek-rul/experiments/punet_matrix/*/f*_s*/preds_point.csv 2>/dev/null | wc -l)/60"
tail -2 ~/deepseek-rul/logs/matrix.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
