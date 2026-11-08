#!/bin/bash
pkill -9 -f run_e2e_cqr 2>/dev/null
pkill -9 -f run_qr_raw 2>/dev/null
sleep 3
echo "after kill: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
cd ~/deepseek-rul
rm -f experiments/punet_e2e/*/f0_s42/e2e_qr_raw_none.json
nohup bash ~/run_qr_raw.sh > ~/deepseek-rul/logs/qrraw_all.log 2>&1 &
sleep 10
echo "after launch: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
