#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo "trainers=$(pgrep -c -f '[r]un_candidate')"
cat /home/yndx/p8/cp.state 2>/dev/null
ls experiments/cplabels/FD002/*/metrics.json 2>/dev/null | wc -l
tail -n 2 $(ls -t logs/order03/cp_*.log 2>/dev/null | head -1) 2>/dev/null | cut -c1-70
