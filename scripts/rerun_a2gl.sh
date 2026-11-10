#!/bin/bash
# Rerun the two FD002 A2-GL runs with the corrected global-shrinkage penalty.
set -u
cd ~/deepseek-rul || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH:-}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
OUT=experiments/tim_sprint
for SD in 42 7; do
  $PY scripts/run_a2.py --subset FD002 --fold 0 --seed "$SD" --arm A2-GL --out "$OUT" \
      >> "logs/a2gl_${SD}.log" 2>&1 &
done
wait
echo "$(date +%F" "%T) A2GL_FD002_DONE" >> logs/a2gl.log
