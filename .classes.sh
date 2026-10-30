#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
grep -n '^class ' src/candidates/model.py
