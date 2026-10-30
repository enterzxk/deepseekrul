#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
grep -n '"A"\|"base"\|variant ==' src/candidates/model.py scripts/run_candidate.py | head -20
