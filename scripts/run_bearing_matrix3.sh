#!/bin/bash
# Corrected bearing LOBO: 17 bearings x 4 arms (A / C / C-Fixed / C-MLP), 3-way parallel.
set -u
cd ~/deepseek-rul || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH:-}
export PY=/home/yndx/miniconda3/envs/a40_env/bin/python
OUT=experiments/bearing_tim2
mkdir -p "$OUT" logs

run_one() {
  B=$1; ARM=$2
  local D="$OUT/$ARM/$B.json"
  if [ -f "$D" ]; then
    echo "$(date +%F" "%T) skip $B $ARM" >> logs/bearing2c.log
    return 0
  fi
  $PY scripts/run_bearing2.py --arm "$ARM" --holdout "$B" --out "$OUT" >> "logs/b2c_${B}_${ARM}.log" 2>&1
  local rc=$?
  echo "$(date +%F" "%T) $B $ARM rc=$rc" >> logs/bearing2c.log
  return 0
}
export -f run_one
export PY OUT LD_LIBRARY_PATH

: > tasks_b2c.txt
for B in Lear_Bearing1_1 Lear_Bearing1_2 Lear_Bearing2_1 Lear_Bearing2_2 Lear_Bearing3_1 Lear_Bearing3_2 \
         Full_Bearing1_3 Full_Bearing1_4 Full_Bearing1_5 Full_Bearing1_6 Full_Bearing1_7 \
         Full_Bearing2_3 Full_Bearing2_4 Full_Bearing2_5 Full_Bearing2_6 Full_Bearing2_7 Full_Bearing3_3; do
  for ARM in A C C-Fixed C-MLP; do
    echo "$B $ARM" >> tasks_b2c.txt
  done
done
echo "$(date +%F" "%T) START $(wc -l < tasks_b2c.txt) runs" >> logs/bearing2c.log
cat tasks_b2c.txt | xargs -P 3 -L 1 bash -c 'run_one $0 $1'
echo "$(date +%F" "%T) ALL_DONE" >> logs/bearing2c.log
