#!/bin/bash
# old workers lost their parent -> ppid 1. Kill them to avoid racing on the same cell.
for pid in $(ps -eo pid,ppid,cmd | awk '$2==1 && /run_e2e_cqr.py/ {print $1}'); do
  kill -9 $pid 2>/dev/null && echo "killed orphan $pid"
done
sleep 3
echo "python procs now: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
ps -eo pid,cmd | grep '[r]un_e2e_cqr.py' | sed 's/.*--subset //' | cut -d' ' -f1 | sort | uniq -c
