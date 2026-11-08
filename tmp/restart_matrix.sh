#!/bin/bash
pkill -9 -f "run_e2e_cqr.py --subset" 2>/dev/null
pkill -9 -f run_matrix.sh 2>/dev/null
sleep 3
cd ~/deepseek-rul
rm -rf experiments/punet_matrix
nohup bash ~/run_matrix.sh > logs/matrix.log 2>&1 &
sleep 15
echo "procs: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
tail -2 logs/matrix.log
