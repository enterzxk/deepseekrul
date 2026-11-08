#!/bin/bash
# kill every previous runner (parents included) to avoid zombie loops
pkill -9 -f run_e2e_cqr 2>/dev/null
pkill -9 -f run_point_5fold 2>/dev/null
pkill -9 -f run_one_subset 2>/dev/null
sleep 3
echo "surviving: $(ps -eo cmd | grep -c '[r]un_e2e_cqr')"

cd ~/deepseek-rul
for S in FD002 FD003 FD004; do
  bash ~/run_one_subset.sh $S > logs/point_$S.log 2>&1 &
  echo "started $S"
done
wait
echo ALL_DONE
