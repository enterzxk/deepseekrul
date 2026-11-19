#!/bin/bash
# Phase 2 M_front: frozen M0 checkpoint + static RHSE-TSH, 4 subsets x 5 seeds x 4 faults.
set -u
cd "$HOME/sprint24" || exit 1
P=/home/user/miniconda3/envs/ncmapss/bin/python
export PYTHONPATH="$HOME/sprint24"
OUT="$HOME/sprint24/experiments/phase2"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
SUBSETS=(FD001 FD002 FD003 FD004)
SEEDS=(101 202 303 404 505)
FAULTS=(bias drift stuck noise)

for s in "${SUBSETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for f in "${FAULTS[@]}"; do
      log="$LOGS/mfront_${s}_s${seed}_${f}.log"
      if [[ -f "$log" ]] && grep -q "R3" "$log"; then
        echo "SKIP $s $seed $f"
        continue
      fi
      echo "=== RUN mfront_${s}_s${seed}_${f} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$log"
      "$P" -m paper_b.modules.phase2_front --subset "$s" --seed "$seed" \
        --fault "$f" --mag 3 --sensors 0,5,10 --gamma 1.30 --onset 0.5 >> "$log" 2>&1
      echo "=== END rc=$? ===" | tee -a "$log"
    done
  done
done
echo "MFRONT QUEUE DONE"
