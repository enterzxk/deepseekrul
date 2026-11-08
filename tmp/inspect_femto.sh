#!/bin/bash
cd ~/femto
echo "--- Learning_set ---"
ls Learning_set | head -5
echo "count: $(ls Learning_set | wc -l)"
echo "--- one bearing ---"
B=$(ls Learning_set | head -1)
ls Learning_set/$B | head -4
echo "files: $(ls Learning_set/$B | wc -l)"
echo "--- csv head ---"
head -2 Learning_set/$B/$(ls Learning_set/$B | head -1)
echo "rows: $(wc -l < Learning_set/$B/$(ls Learning_set/$B | head -1))"
echo "--- seeds job ---"
echo "procs: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
ls ~/deepseek-rul/experiments/punet_seeds 2>/dev/null | head -4
