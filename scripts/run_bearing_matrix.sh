#!/bin/bash
# Bearing LOBO matrix: 6 Learning bearings x 3 arms (A direct / C generic gate / D consistency gate)
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
    echo "$(date +%F" "%T) skip $B $ARM" >> logs/bearing.log
    return 0
  fi
  $PY scripts/run_bearing.py --arm "$ARM" --holdout "$B" --out "$OUT" >> "logs/bearing_${B}_${ARM}.log" 2>&1
  local rc=$?
  echo "$(date +%F" "%T) $B $ARM rc=$rc" >> logs/bearing.log
  return 0
}
export -f run_one
export PY OUT LD_LIBRARY_PATH

: > tasks_bearing.txt
for B in Lear_Bearing1_1 Lear_Bearing1_2 Lear_Bearing2_1 Lear_Bearing2_2 Lear_Bearing3_1 Lear_Bearing3_2; do
  for ARM in A C D; do
    echo "$B $ARM" >> tasks_bearing.txt
  done
done
echo "$(date +%F" "%T) START $(wc -l < tasks_bearing.txt) runs" >> logs/bearing.log
cat tasks_bearing.txt | xargs -P 3 -L 1 bash -c 'run_one $0 $1'
echo "$(date +%F" "%T) ALL_DONE" >> logs/bearing.log
