#!/bin/bash
echo "=== seeds job ==="
echo "procs: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
ls ~/deepseek-rul/experiments/punet_seeds 2>/dev/null | head -4
echo "=== femto features ==="
tail -3 ~/deepseek-rul/logs/femto_feat.log 2>/dev/null
ls ~/deepseek-rul/data/femto_features 2>/dev/null | wc -l
