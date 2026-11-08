#!/bin/bash
echo "--- matrix procs ---"
ps -eo pid,etime,pcpu,cmd | grep '[r]un_e2e_cqr' | cut -c1-95
echo "--- worker parents ---"
ps -eo pid,cmd | grep '[m]atrix2seeds' | cut -c1-60
echo "--- log tail ---"
tail -6 ~/deepseek-rul/logs/matrix2.log | cut -c1-70
echo "--- newest json ---"
ls -t ~/deepseek-rul/experiments/punet_matrix/*/*/e2e_*_none.json 2>/dev/null | head -3 | xargs -r ls -l --time-style=+%H:%M | awk '{print $6, $7}'
echo "--- now ---"; date +%H:%M
echo "--- femto ---"; tail -1 ~/deepseek-rul/logs/femto6.log
