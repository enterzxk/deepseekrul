#!/bin/bash
# B2 dev-set matrix: 2 arms x 2 seeds on FD002, 2-way parallel (A2 matrix keeps the other slots).
set -u
cd ~/deepseek-rul || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH:-}
export PY=/home/yndx/miniconda3/envs/a40_env/bin/python
OUT=experiments/tim_sprint
mkdir -p "$OUT" logs

run_one() {
  S=$1; F=$2; SD=$3; ARM=$4
  local D="$OUT/$S/f${F}_s${SD}_${ARM}"
  [ -f "$D/metrics.json" ] && { echo "$(date +%F' '%T) skip $S f$F s$SD $ARM" >> logs/b2_matrix.log; return 0; }
  $PY scripts/run_b2.py --subset "$S" --fold "$F" --seed "$SD" --arm "$ARM" \
      --out "$OUT" >> "logs/b2_${S}_f${F}_s${SD}_${ARM}.log" 2>&1
  echo "$(date +%F' '%T) $S f$F s$SD $ARM rc=$?" >> logs/b2_matrix.log
}
export -f run_one
export PY OUT LD_LIBRARY_PATH

: > tasks_b2.txt
for SD in 42 7; do
  for ARM in B2 B2-G; do
    echo "FD002 0 $SD $ARM" >> tasks_b2.txt
  done
done
echo "$(date +%F' '%T) START $(wc -l < tasks_b2.txt) runs" >> logs/b2_matrix.log
cat tasks_b2.txt | xargs -P 2 -L 1 bash -c 'run_one $0 $1 $2 $3'
echo "$(date +%F' '%T) ALL_DONE" >> logs/b2_matrix.log
