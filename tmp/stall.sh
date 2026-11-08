#!/bin/bash
echo "procs: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
echo "bash workers: $(ps -eo cmd | grep -c '[m]atrix2seeds')"
echo "--- running ---"
ps -eo etime,cmd | grep '[r]un_e2e_cqr' | cut -c1-70
echo "--- log tail ---"
tail -4 ~/deepseek-rul/logs/matrix2.log | cut -c1-80
echo "--- newest json ---"
ls -t ~/deepseek-rul/experiments/punet_matrix/*/*/e2e_*_none.json 2>/dev/null | head -2 | xargs -r ls -l --time-style=+%H:%M | awk '{print $6, $7}'
date +%H:%M
