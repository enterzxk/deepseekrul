#!/bin/bash
echo "FD004 files: $(ls ~/deepseek-rul/experiments/punet_e2e/FD004/f*_s42/e2e_point_none.json 2>/dev/null | wc -l)"
echo "--- fd004 log tail ---"
tail -4 ~/deepseek-rul/logs/point_FD004.log | cut -c1-200
echo "--- all procs ---"
ps -eo pid,etime,cmd | grep "[r]un_e2e_cqr.py" | cut -c1-100
echo "--- orchestrate ---"
ps -eo pid,etime,cmd | grep "[o]rchestrate.sh" | cut -c1-80
