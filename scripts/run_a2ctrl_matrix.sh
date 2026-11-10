#!/bin/bash
# A2 control matrix (Codex q8): real event penalty vs pseudo-event vs global shrinkage.
# A2 (real) is already trained; this runs A2-PS and A2-GL on FD002/FD004 x seed 42/7 = 8 runs.
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
    echo "$(date +%F" "%T) skip $S f$F s$SD $ARM" >> logs/a2ctrl.log
    return 0
  fi
  $PY scripts/run_a2.py --subset "$S" --fold "$F" --seed "$SD" --arm "$ARM" \
      --out "$OUT" >> "logs/a2ctrl_${S}_f${F}_s${SD}_${ARM}.log" 2>&1
  local rc=$?
  echo "$(date +%F" "%T) $S f$F s$SD $ARM rc=$rc" >> logs/a2ctrl.log
  return 0
}
export -f run_one
export PY OUT LD_LIBRARY_PATH

: > tasks_a2ctrl.txt
for S in FD002 FD004; do
  for SD in 42 7; do
    for ARM in A2-PS A2-GL; do
      echo "$S 0 $SD $ARM" >> tasks_a2ctrl.txt
    done
  done
done
echo "$(date +%F" "%T) START $(wc -l < tasks_a2ctrl.txt) runs" >> logs/a2ctrl.log
cat tasks_a2ctrl.txt | xargs -P 4 -L 1 bash -c 'run_one $0 $1 $2 $3'
echo "$(date +%F" "%T) ALL_DONE" >> logs/a2ctrl.log
