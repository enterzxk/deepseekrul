#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
grep -n '"config":' -A 6 scripts/run_candidate.py | head -12
