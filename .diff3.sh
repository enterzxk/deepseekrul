#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
diff -u scripts/frozen/levelb_runner_8fce61ff.py scripts/run_candidate.py > /tmp/rundiff.txt 2>&1
echo LINES=$(wc -l < /tmp/rundiff.txt)
head -c 3000 /tmp/rundiff.txt
