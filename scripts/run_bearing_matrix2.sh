#!/bin/bash
# Extend the bearing LOBO to the 11 Full_Test bearings, arms C and D only.
set -u
cd ~/deepseek-rul || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH:-}
export PY=/home/yndx/miniconda3/envs/a40_env/bin/python
OUT=experiments/bearing_tim
mkdir -p "$OUT" logs

run_one() {
  B=$1; ARM=$2
  local D="$OUT/$ARM/$B.json"
  if [ -f "$D" ]; then
    echo "$(date +%F" "%T) skip $B $ARM" >> logs/bearing2.log
    return 0
  fi
  $PY scripts/run_bearing.py --arm "$ARM" --holdout "$B" --out "$OUT" >> "logs/bearing2_${B}_${ARM}.log" 2>&1
  local rc=$?
  echo "$(date +%F" "%T) $B $ARM rc=$rc" >> logs/bearing2.log
  return 0
}
export -f run_one
export PY OUT LD_LIBRARY_PATH

: > tasks_bearing2.txt
for B in Full_Bearing1_3 Full_Bearing1_4 Full_Bearing1_5 Full_Bearing1_6 Full_Bearing1_7 \
         Full_Bearing2_3 Full_Bearing2_4 Full_Bearing2_5 Full_Bearing2_6 Full_Bearing2_7 Full_Bearing3_3; do
  for ARM in C D; do
    echo "$B $ARM" >> tasks_bearing2.txt
  done
done
echo "$(date +%F" "%T) START $(wc -l < tasks_bearing2.txt) runs" >> logs/bearing2.log
cat tasks_bearing2.txt | xargs -P 3 -L 1 bash -c 'run_one $0 $1'
echo "$(date +%F" "%T) ALL_DONE" >> logs/bearing2.log
