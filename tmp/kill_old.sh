#!/bin/bash
for pid in $(ps -eo pid,cmd | grep 'punet_seeds' | grep -v grep | awk '{print $1}'); do
  kill -9 $pid 2>/dev/null && echo "killed python $pid"
done
pkill -9 -f run_seeds.sh 2>/dev/null
sleep 2
echo "matrix: $(ps -eo cmd | grep -c 'punet_matrix')  seeds: $(ps -eo cmd | grep -c 'punet_seeds')"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
