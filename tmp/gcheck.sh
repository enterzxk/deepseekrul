#!/bin/bash
echo "=== seeds ==="
echo "cqr $(ls ~/deepseek-rul/experiments/punet_seeds/*/f*_s*/e2e_cqr_none.json 2>/dev/null | wc -l)/60  point $(ls ~/deepseek-rul/experiments/punet_seeds/*/f*_s*/e2e_point_none.json 2>/dev/null | wc -l)/60  qr $(ls ~/deepseek-rul/experiments/punet_seeds/*/f*_s*/e2e_qr_raw_none.json 2>/dev/null | wc -l)/60"
echo "=== femto ==="
tail -3 ~/deepseek-rul/logs/femto_exp.log
echo "femto proc: $(ps -eo cmd | grep -c '[r]un_femto_cqr')"
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
