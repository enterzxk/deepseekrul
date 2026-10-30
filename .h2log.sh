#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
ls -t logs/order03/m7_s8_*.log logs/order03/m7_s9_*.log 2>/dev/null | head -3
f=$(ls -t logs/order03/m7_s*_*.log 2>/dev/null | head -1)
echo "--- $f"; tail -n 6 "$f"
echo '--- gpu'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -2
