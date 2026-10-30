#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo trainers=$(pgrep -c -f '[r]un_candidate') done=$(ls experiments/q025/FD002/*/metrics.json 2>/dev/null | wc -l)
cat /home/yndx/p8/q.state 2>/dev/null
tail -n 1 $(ls -t logs/order03/q025_*.log 2>/dev/null | head -1) 2>/dev/null | cut -c1-60
