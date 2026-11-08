#!/bin/bash
pkill -9 -f "heads point --save-ckpt" 2>/dev/null
pkill -9 -f run_mcd.sh 2>/dev/null
sleep 2
echo "procs now: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
