#!/bin/bash
pkill -9 -f run_femto_cqr 2>/dev/null
sleep 2
echo "femto procs: $(ps -eo cmd | grep -c '[r]un_femto_cqr')"
echo "seeds procs: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
