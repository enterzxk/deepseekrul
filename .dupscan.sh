#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
find experiments -name protocol_c_predictions.csv 2>/dev/null | xargs md5sum 2>/dev/null | sort > /tmp/hashes.txt
echo "total files: $(wc -l < /tmp/hashes.txt)"
echo "unique hashes: $(cut -d' ' -f1 /tmp/hashes.txt | sort -u | wc -l)"
echo "--- duplicate groups ---"
cut -d' ' -f1 /tmp/hashes.txt | sort | uniq -d | while read h; do
  echo "HASH $h"
  grep "^$h" /tmp/hashes.txt | awk '{print "   " $2}'
done | head -40
