#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi/results/dsh/review_r3/sources || exit 1
for f in p2607_24809 patchtst; do
  pdftotext -layout $f.pdf $f.txt 2>/dev/null
  echo "$f: $(wc -c < $f.txt) chars, pages=$(python3 -c "import sys;print(open('$f.txt',encoding='utf-8',errors='ignore').read().count(chr(12))+1)")"
done
ls -l *.pdf *.txt | awk '{print $5, $9}'
