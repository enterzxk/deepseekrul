#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
echo '--- FD002 roots with matched preds:'; 
ls -d experiments/*/*/FD002 2>/dev/null | sed 's|/FD002||' | sort | head -30
echo '--- count per cell:';
for c in M1_global M6_tfA M0_raw_tf; do
  n=$(ls -d experiments/*/$c/FD002/*/ 2>/dev/null | wc -l)
  echo "$c: $n run dirs"
done
