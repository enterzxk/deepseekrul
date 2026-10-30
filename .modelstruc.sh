#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
grep -n 'class CandidateModel' -A 30 src/candidates/model.py | head -45
