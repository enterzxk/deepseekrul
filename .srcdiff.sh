#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo '=== files in frozen_src:'; ls scripts/frozen/frozen_src 2>/dev/null | head
echo '=== diff dataset.py'; diff -u scripts/frozen/frozen_src/dataset.py src/dataset.py > /tmp/d1.txt 2>&1; echo lines=$(wc -l < /tmp/d1.txt); grep -E '^[-+][^-+]' /tmp/d1.txt | head -25
echo '=== diff model'; diff -u scripts/frozen/frozen_src/candidates/model.py src/candidates/model.py > /tmp/d2.txt 2>&1; echo lines=$(wc -l < /tmp/d2.txt); grep -E '^[-+][^-+]' /tmp/d2.txt | head -30
