#!/bin/bash
echo "procs: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
tail -2 ~/deepseek-rul/logs/seeds_all.log 2>/dev/null
ls ~/deepseek-rul/experiments/punet_seeds 2>/dev/null | head -3
