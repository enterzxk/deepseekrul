#!/bin/bash
echo "--- running python procs ---"
ps -eo pid,etime,pcpu,cmd | grep '[r]un_e2e_cqr.py' | cut -c1-95
echo "--- recent completions (json mtimes) ---"
ls -lt --time-style=+%H:%M ~/deepseek-rul/experiments/punet_matrix/*/f*_s*/e2e_cqr_none.json 2>/dev/null | head -6 | awk '{print $6, $7}'
date +%H:%M
