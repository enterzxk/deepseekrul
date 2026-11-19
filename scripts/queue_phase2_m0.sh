#!/bin/bash
# Phase 2 M0: DSPT-V0 (V0C) clean baseline, 4 subsets x 5 seeds.
# Serial, idempotent, test sealed. Run from ~/sprint24.
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
    rid="phase2_M0_${s}_s${seed}"
    d="$OUT/$rid"
    log="$LOGS/$rid.log"
    if is_done "$d"; then
      echo "SKIP $rid"
      continue
    fi
    echo "=== RUN $rid $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$log"
    "$P" -m paper_b.hc.run_v0 --variant V0C --subset "$s" --seed "$seed" \
      --run-id "$rid" --hypothesis Phase2_M0 --data-dir "$DATA" \
      --out-root "$OUT" --code-lock "$LOCK" >> "$log" 2>&1
    rc=$?
    echo "=== END $rid rc=$rc ===" | tee -a "$log"
  done
done
echo "M0 QUEUE DONE"
