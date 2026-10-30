#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo '--- split8 tree'; find experiments/split8 -maxdepth 4 -type d 2>/dev/null | head -10
echo '--- any metrics'; find experiments/split8 experiments/split9 -name 'metrics.json' 2>/dev/null | head -8
echo '--- last log lines'; f=$(ls -t logs/order03/m7_s9_*.log | head -1); tail -n 8 "$f"
