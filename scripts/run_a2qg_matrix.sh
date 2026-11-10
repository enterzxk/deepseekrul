#!/bin/bash
# Quality-gated A2: FD002/FD004 x seed 42/7 x {A2-QG, A2-QGg} = 8 runs.
set -u
cd ~/deepseek-rul || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH:-}
export PY=/home/yndx/miniconda3/envs/a40_env/bin/python
OUT=experiments/tim_sprint
mkdir -p "$OUT" logs

run_one() {
  S=$1; F=$2; SD=$3; ARM=$4
  local D="$OUT/$S/f${F}_s${SD}_${ARM}"
  if [ -f "$D/metrics.json" ]; then
    echo "$(date +%F" "%T) skip $S f$F s$SD $ARM" >> logs/a2qg.log
    return 0
  fi
  $PY scripts/run_a2.py --subset "$S" --fold "$F" --seed "$SD" --arm "$ARM" \
      --out "$OUT" >> "logs/a2qg_${S}_f${F}_s${SD}_${ARM}.log" 2>&1
  local rc=$?
  echo "$(date +%F" "%T) $S f$F s$SD $ARM rc=$rc" >> logs/a2qg.log
  return 0
}
export -f run_one
export PY OUT LD_LIBRARY_PATH

: > tasks_a2qg.txt
for S in FD002 FD004; do
  for SD in 42 7; do
    for ARM in A2-QG A2-QGg; do
      echo "$S 0 $SD $ARM" >> tasks_a2qg.txt
    done
  done
done
echo "$(date +%F" "%T) START $(wc -l < tasks_a2qg.txt) runs" >> logs/a2qg.log
cat tasks_a2qg.txt | xargs -P 4 -L 1 bash -c 'run_one $0 $1 $2 $3'
echo "$(date +%F" "%T) ALL_DONE" >> logs/a2qg.log
