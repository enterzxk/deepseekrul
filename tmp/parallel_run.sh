#!/bin/bash
pkill -9 -f run_e2e_cqr 2>/dev/null
sleep 2
for S in FD002 FD003 FD004; do
  nohup bash ~/run_one_subset.sh $S > ~/deepseek-rul/logs/point_$S.log 2>&1 &
  echo "launched $S pid $!"
done
sleep 8
ps -eo pid,cmd | grep "[r]un_e2e_cqr" | wc -l
