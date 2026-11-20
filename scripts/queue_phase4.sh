#!/bin/bash
# Phase 4: R3 vs fair naive R5b/R5m, FD001/FD003 x bias/drift/stuck x onset 0.2/0.5 x 3 seeds.
set -u
cd "$HOME/sprint24" || exit 1
P=/home/user/miniconda3/envs/ncmapss/bin/python
export PYTHONPATH="$HOME/sprint24"
OUT="$HOME/sprint24/experiments/phase4"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

root_for() {
  case "$1" in
    FD003) echo "$HOME/sprint24/experiments/hc/d80" ;;
    *) echo "$HOME/sprint24/experiments/hc/_gate" ;;
  esac
}

for subset in FD001 FD003; do
  for fault in bias drift stuck; do
    for onset in 0.2 0.5; do
      for seed in 101 202 303; do
        tag="${subset}_${fault}_onset${onset}_s${seed}"
        log="$LOGS/$tag.log"
        if [[ -f "$log" ]] && grep -q "R5m" "$log"; then
          echo "SKIP $tag"
          continue
        fi
        echo "=== RUN $tag $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$log"
        "$P" -m paper_b.modules.phase4_r5b --subset "$subset" --seed "$seed" \
          --root "$(root_for "$subset")" --fault "$fault" --onset "$onset" \
          --mag 3 --sensors 0,5,10 >> "$log" 2>&1
        echo "=== END rc=$? ===" | tee -a "$log"
      done
    done
  done
done
echo "PHASE4 DONE"
