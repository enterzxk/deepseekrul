#!/bin/bash
ps -eo pid,etime,cmd | grep '[r]un_e2e_cqr.py' | sed 's/.*--out //' | awk '{print $1}' | sort | uniq -c
echo "--- total ---"
ps -eo cmd | grep -c '[r]un_e2e_cqr.py'
