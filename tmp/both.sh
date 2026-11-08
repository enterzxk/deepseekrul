#!/bin/bash
echo "--- femto ---"
ps -eo pid,etime,pcpu,cmd | grep '[r]un_femto_cqr' | cut -c1-80
tail -2 ~/deepseek-rul/logs/femto_gpu.log
echo "--- matrix ---"
ps -eo etime,cmd | grep '[r]un_e2e_cqr' | cut -c1-55
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
