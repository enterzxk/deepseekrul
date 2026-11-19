#!/bin/bash
# Phase 2 M0+Aug: static 30% training-window augmentation, 4 subsets x 5 seeds.
# Runs on cuda:1 so it can proceed in parallel with M0 on cuda:0.
set -u
cd "$HOME/sprint24" || exit 1
P=/home/user/miniconda3/envs/ncmapss/bin/python
export PYTHONPATH="$HOME/sprint24"
OUT="$HOME/sprint24/experiments/phase2"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
SUBSETS=(FD001 FD002 FD003 FD004)
SEEDS=(101 202 303 404 505)
LOCK="docs/paper_b_hc/registry/code_lock_phase2.json"
DATA="/home/user/rulb/data/CMAPSS"

is_done() {
  local d="$1"
  [[ -f "$d/metrics.json" && -f "$d/close.json" ]]
}

for s in "${SUBSETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    rid="phase2_M0Aug_${s}_s${seed}"
    d="$OUT/$rid"
    log="$LOGS/$rid.log"
    if is_done "$d"; then
      echo "SKIP $rid"
      continue
    fi
    echo "=== RUN $rid $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$log"
    "$P" -m paper_b.modules.phase2_aug --variant V0C --subset "$s" --seed "$seed" \
      --run-id "$rid" --hypothesis Phase2_M0Aug --data-dir "$DATA" \
      --out-root "$OUT" --code-lock "$LOCK" --device cuda:1 >> "$log" 2>&1
    rc=$?
    echo "=== END $rid rc=$rc ===" | tee -a "$log"
  done
done
echo "M0AUG QUEUE DONE"
