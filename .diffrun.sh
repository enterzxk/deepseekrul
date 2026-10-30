#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
ls scripts/frozen/ 2>/dev/null | head
echo "--- sizes"
wc -l scripts/frozen/levelb_runner_8fce61ff.py scripts/run_candidate.py 2>/dev/null
echo "--- diff summary (first 60 changed hunks)"
diff <(grep -v "^\s*#" scripts/frozen/levelb_runner_8fce61ff.py) <(grep -v "^\s*#" scripts/run_candidate.py) | head -80
