#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo '--- subset handling in runner'; grep -n 'subset' scripts/run_candidate.py | head -12
echo '--- load function signature'; grep -n 'def load_cmapss_train' -A 6 src/dataset.py
