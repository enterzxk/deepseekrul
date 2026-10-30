#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo '--- FD001 cells:'; ls -d experiments/*/*/FD001 2>/dev/null | sed 's|experiments/||' | head -20
