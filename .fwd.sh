#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
grep -n 'def forward' -A 12 src/candidates/model.py | head -30
echo '--- return keys'; grep -n "return {" -A 4 src/candidates/model.py | head -12
