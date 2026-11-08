#!/bin/bash
echo "--- gpu apps ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "--- load ---"; cat /proc/loadavg
echo "--- our runs ---"
ps -eo pid,etime,pcpu,rss,cmd | grep '[r]un_e2e_cqr' | awk '{print $1, $2, $3, int($4/1024)"MB"}' 
echo "--- mem ---"; free -g | head -2
echo "--- io wait ---"; top -bn1 | head -3 | tail -1
