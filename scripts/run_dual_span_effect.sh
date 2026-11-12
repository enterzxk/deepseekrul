#!/usr/bin/env bash
set -u
source /media/sangfor/vdb/user/zxk/env.sh
cd "$HC_ROOT"
"$PY" -m paper_b.experiments.audit_dual_span_effect \
  --run-dir /tmp/audit_runs/A8_base_L70/FD001/s101 \
  --data-dir data/CMAPSS --device cuda:0 \
  --out /tmp/dual_span_effect.json > logs/dual_span_effect.log 2>&1
echo "ablate_exit=$?"
cat logs/dual_span_effect.log
echo "--- grid2 ---"
grep -c "^DONE" logs/vanilla_features2.log
grep "^DONE" logs/vanilla_features2.log | tail -1
