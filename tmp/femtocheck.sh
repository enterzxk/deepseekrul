#!/bin/bash
ps -eo pid,etime,pcpu,cmd | grep '[r]un_femto_cqr' | cut -c1-100
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- counts ---"
echo "femto log lines: $(wc -l < ~/deepseek-rul/logs/femto_exp.log)"
