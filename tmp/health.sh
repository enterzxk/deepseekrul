#!/bin/bash
echo "--- python procs ---"
ps -eo pid,etime,pcpu,cmd | grep "[r]un_e2e_cqr.py" | cut -c1-110
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- logs ---"
for S in FD002 FD003 FD004; do
  echo "[$S] $(tail -1 ~/deepseek-rul/logs/point_$S.log 2>/dev/null | cut -c1-80)"
done
