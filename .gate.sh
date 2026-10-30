#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
grep -n 'self.gate\|alpha' src/candidates/model.py | head -12
