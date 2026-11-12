#!/usr/bin/env bash
set -u
source /media/sangfor/vdb/user/zxk/env.sh
cd "$HC_ROOT"
"$PY" -m paper_b.experiments.audit_dspt_base --out /tmp/audit_dspt_base.json \
  > logs/audit_dspt_base.log 2>&1
echo "audit_exit=$?"
tail -8 logs/audit_dspt_base.log
echo "--- grid progress ---"
tail -3 logs/vanilla_missing.log
