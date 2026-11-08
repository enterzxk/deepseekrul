#!/bin/bash
for S in FD002 FD003 FD004; do
  echo "=== $S ==="
  tail -2 ~/deepseek-rul/logs/point_$S.log | cut -c1-160
done
echo "--- procs ---"
ps -eo pid,etime,pcpu,cmd | grep "[r]un_e2e_cqr.py" | cut -c1-90
