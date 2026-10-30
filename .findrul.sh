#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
grep -rn "def build_piecewise_linear_rul" src/ scripts/ 2>/dev/null
echo "---"
sed -n '40,50p' scripts/run_candidate.py
