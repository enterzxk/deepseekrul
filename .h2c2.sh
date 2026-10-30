#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo trainers=$(pgrep -c -f '[r]un_candidate')
cat /home/yndx/p8/h2.state 2>/dev/null
echo s8=$(ls experiments/split8/base/FD002/*/metrics.json 2>/dev/null | wc -l) s9=$(ls experiments/split9/base/FD002/*/metrics.json 2>/dev/null | wc -l)
