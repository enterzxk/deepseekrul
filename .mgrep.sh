#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo '=== FROZEN matched block'; grep -n 'matched' scripts/frozen/levelb_runner_8fce61ff.py | head -20
echo '=== CURRENT matched block'; grep -n 'matched' scripts/run_candidate.py | head -20
