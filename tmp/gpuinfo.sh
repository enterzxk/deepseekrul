#!/bin/bash
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "--- processes on gpu ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -12
echo "--- our procs ---"
ps -eo cmd | grep -c '[r]un_e2e_cqr.py'
