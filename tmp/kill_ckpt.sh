#!/bin/bash
for pid in $(ps -eo pid,cmd | grep 'save-ckpt' | grep -v grep | awk '{print $1}'); do
  kill -9 $pid 2>/dev/null && echo "killed $pid"
done
sleep 2
echo "remaining run_e2e_cqr: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
ps -eo cmd | grep '[r]un_e2e_cqr.py' | sed 's/.*--heads //' | cut -d' ' -f1 | sort | uniq -c
