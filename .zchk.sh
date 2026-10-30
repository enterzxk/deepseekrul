#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo '=== _z definition'; grep -n '_z\b\|def _z' scripts/run_candidate.py | head
echo '=== context 585-615'; sed -n '585,615p' scripts/run_candidate.py
