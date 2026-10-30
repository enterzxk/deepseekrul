#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
sed -n '50,95p' scripts/run_candidate.py
