#!/bin/bash
echo "cqr dumps: $(ls ~/deepseek-rul/experiments/punet_matrix/*/f*_s*/preds_cqr.csv 2>/dev/null | wc -l)"
echo "point dumps: $(ls ~/deepseek-rul/experiments/punet_matrix/*/f*_s*/preds_point.csv 2>/dev/null | wc -l)"
echo "--- last 3 log lines ---"
tail -3 ~/deepseek-rul/logs/matrix.log | cut -c1-90
echo "--- log mtime ---"
ls -l --time-style=+%H:%M ~/deepseek-rul/logs/matrix.log | awk '{print $6}'
date +%H:%M
