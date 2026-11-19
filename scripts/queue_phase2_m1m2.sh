#!/bin/bash
# Phase 2 M1/M2: RHSE-DSPT clean and +aug, 4 subsets x 5 seeds each.
set -u
cd "$HOME/sprint24" || exit 1
P=/home/user/miniconda3/envs/ncmapss/bin/python
export PYTHONPATH="$HOME/sprint24"
OUT="$HOME/sprint24/experiments/phase2"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
SUBSETS=(FD001 FD002 FD003 FD004)
SEEDS=(101 202 303 404 505)

for arm in M1 M2; do
  for s in "${SUBSETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      d="$OUT/phase2_${arm}_${s}_s${seed}"
      log="$LOGS/${arm}_${s}_s${seed}.log"
      if [[ -f "$d/metrics.json" ]]; then
        echo "SKIP $arm $s $seed"
        continue
      fi
      echo "=== RUN ${arm}_${s}_s${seed} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$log"
      "$P" -m paper_b.modules.phase2_train --subset "$s" --seed "$seed" \
        --arm "$arm" --device cuda:1 --out-root "$OUT" >> "$log" 2>&1
      echo "=== END rc=$? ===" | tee -a "$log"
    done
  done
done
echo "M1M2 QUEUE DONE"
