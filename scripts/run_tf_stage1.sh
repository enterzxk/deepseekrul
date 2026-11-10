#!/bin/bash
# Stage 1 screening: 6 representative bearings x {T1,T2,T3}, 4-way parallel.
set -u
cd ~/deepseek-rul || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH:-}
export PY=/home/yndx/miniconda3/envs/a40_env/bin/python
OUT=experiments/bearing_tf
mkdir -p "$OUT" logs

run_one() {
  B=$1; A=$2
  local D="$OUT/$A/$B.json"
  if [ -f "$D" ]; then
    echo "$(date +%F" "%T) skip $B $A" >> logs/tf.log
    return 0
  fi
  $PY scripts/run_bearing_tf.py --arch "$A" --holdout "$B" --out "$OUT" >> "logs/tf_${B}_${A}.log" 2>&1
  local rc=$?
  echo "$(date +%F" "%T) $B $A rc=$rc" >> logs/tf.log
  return 0
}
export -f run_one
export PY OUT LD_LIBRARY_PATH

: > tasks_tf.txt
for B in Lear_Bearing1_1 Lear_Bearing2_1 Lear_Bearing3_1 Full_Bearing1_3 Full_Bearing2_3 Full_Bearing3_3; do
  for A in T1 T2 T3; do
    echo "$B $A" >> tasks_tf.txt
  done
done
echo "$(date +%F" "%T) START $(wc -l < tasks_tf.txt) runs" >> logs/tf.log
cat tasks_tf.txt | xargs -P 4 -L 1 bash -c 'run_one $0 $1'
echo "$(date +%F" "%T) ALL_DONE" >> logs/tf.log
