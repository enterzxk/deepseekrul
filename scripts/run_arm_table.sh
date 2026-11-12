#!/usr/bin/env bash
set -u
source /media/sangfor/vdb/user/zxk/env.sh
cd "$HC_ROOT"
"$PY" -m paper_b.experiments.audit_arm_table \
  --roots paper_b/results/raw_a40,paper_b/results/raw \
  --out paper_b/results/audit_arm_table.csv > logs/audit_arm_table.log 2>&1
echo "exit=$?"
cat logs/audit_arm_table.log
