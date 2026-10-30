#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo '=== scripts referencing test_ or RUL_ :'
grep -rn "test_FD\|RUL_FD\|test_\*\|'test'" scripts/*.py 2>/dev/null | grep -v 'run_candidate\|levelb' | head -10
echo '=== dataset loader reads:'
grep -n 'train_\|test_' src/dataset.py | head -8
