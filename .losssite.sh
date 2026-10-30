#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
grep -n 'soft_nasa_loss\|loss =\|late_weight' scripts/run_candidate.py | head -20
