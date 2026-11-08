#!/bin/bash
cd ~/deepseek-rul/experiments/punet_matrix
echo "--- FD001/f0_s7 ---"; ls -l FD001/f0_s7/ 2>/dev/null | awk '{print $5, $9}'
echo "--- FD003/f2_s42 ---"; ls -l FD003/f2_s42/ 2>/dev/null | awk '{print $5, $9}'
echo "--- all dumps ---"
ls */*/preds_*.csv 2>/dev/null | sed 's|/[^/]*$||' | sort | uniq -c | head -20
