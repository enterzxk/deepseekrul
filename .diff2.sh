#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
diff -u scripts/frozen/levelb_runner_8fce61ff.py scripts/run_candidate.py > /tmp/rundiff.txt 2>&1
echo "diff lines: $(wc -l < /tmp/rundiff.txt)"
grep -E "^[-+][^-+]" /tmp/rundiff.txt | head -60
