#!/bin/bash
# kill the OLD matrix orchestrator (it respawns python children)
pkill -9 -f "bash /home/yndx/run_matrix.sh" 2>/dev/null
pkill -9 -f "run_matrix.sh" 2>/dev/null
sleep 3
echo "python procs: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
ps -eo pid,cmd | grep '[r]un_e2e_cqr.py' | sed 's/.*--subset //' | cut -d' ' -f1 | sort | uniq -c
nvidia-smi --query-gpu=memory.used --format=csv,noheader
