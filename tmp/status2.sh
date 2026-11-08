#!/bin/bash
echo "procs: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "--- qr_raw logs ---"
for S in FD001 FD002 FD003 FD004; do
  echo "[$S] $(grep -c qr_raw ~/deepseek-rul/logs/qrraw_$S.log 2>/dev/null) $(tail -1 ~/deepseek-rul/logs/qrraw_$S.log 2>/dev/null | cut -c1-60)"
done
