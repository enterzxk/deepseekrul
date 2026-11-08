#!/bin/bash
pkill -9 -f run_e2e_cqr 2>/dev/null
pkill -9 -f run_point_5fold 2>/dev/null
pkill -9 -f 'for S in FD001' 2>/dev/null
sleep 3
echo "--- surviving procs ---"
ps -eo pid,cmd | grep -E 'run_e2e_cqr|run_point_5fold' | grep -v grep || echo NONE
cd ~/deepseek-rul
rm -f experiments/punet_e2e/*/f*_s42/e2e_point_none.json
echo "point files after clean: $(ls experiments/punet_e2e/*/f*_s42/e2e_point_none.json 2>/dev/null | wc -l)"
echo "cqr files intact: $(ls experiments/punet_e2e/*/f*_s42/e2e_cqr_none.json 2>/dev/null | wc -l)"
nohup bash ~/run_point_5fold.sh > ~/deepseek-rul/logs/e2e_point_5fold.log 2>&1 &
echo "launched pid $!"
sleep 10
tail -2 ~/deepseek-rul/logs/e2e_point_5fold.log
