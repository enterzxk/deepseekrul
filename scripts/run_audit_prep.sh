#!/usr/bin/env bash
set -u
source /media/sangfor/vdb/user/zxk/env.sh
cd "$HC_ROOT"
"$PY" -m paper_b.experiments.audit_preprocessing --data-dir data/CMAPSS \
  > logs/audit_preprocessing.log 2>&1
echo "exit=$?"
cat logs/audit_preprocessing.log
