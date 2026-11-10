#!/bin/bash
# E0-E3: 17 bearings x 4 feature sets, strict triple-disjoint LOBO, 4-way parallel.
set -u
cd ~/deepseek-rul || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH:-}
export PY=/home/yndx/miniconda3/envs/a40_env/bin/python
OUT=experiments/bearing_tim4
mkdir -p "$OUT" logs

run_one() {
  B=$1; F=$2
  local D="$OUT/$F/$B.json"
  if [ -f "$D" ]; then
    echo "$(date +%F" "%T) skip $B $F" >> logs/e.log
    return 0
  fi
  $PY scripts/run_bearing3.py --feat "$F" --holdout "$B" --out "$OUT" >> "logs/e_${B}_${F}.log" 2>&1
  local rc=$?
  echo "$(date +%F" "%T) $B $F rc=$rc" >> logs/e.log
  return 0
}
export -f run_one
export PY OUT LD_LIBRARY_PATH

: > tasks_e.txt
for B in Lear_Bearing1_1 Lear_Bearing1_2 Lear_Bearing2_1 Lear_Bearing2_2 Lear_Bearing3_1 Lear_Bearing3_2 \
         Full_Bearing1_3 Full_Bearing1_4 Full_Bearing1_5 Full_Bearing1_6 Full_Bearing1_7 \
         Full_Bearing2_3 Full_Bearing2_4 Full_Bearing2_5 Full_Bearing2_6 Full_Bearing2_7 Full_Bearing3_3; do
  for F in none rms hf bands; do
    echo "$B $F" >> tasks_e.txt
  done
done
echo "$(date +%F" "%T) START $(wc -l < tasks_e.txt) runs" >> logs/e.log
cat tasks_e.txt | xargs -P 4 -L 1 bash -c 'run_one $0 $1'
echo "$(date +%F" "%T) ALL_DONE" >> logs/e.log
