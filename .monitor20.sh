#!/bin/bash
# Poll the two E20 batches every 5 minutes until both report ALLDONE (max ~5h).
for i in $(seq 1 60); do
  out=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=20 -o UpdateHostKeys=no lab-server \
    'f3=$(grep -c "^done FD003" /tmp/e20f3_status.txt 2>/dev/null); f3a=$(grep -c ALLDONE /tmp/e20f3_status.txt 2>/dev/null); \
     f4=$(grep -c "^done FD004" /tmp/e20f4_status.txt 2>/dev/null); f4a=$(grep -c ALLDONE /tmp/e20f4_status.txt 2>/dev/null); \
     bad=$(grep -h "rc=[^0]" /tmp/e20f3_status.txt /tmp/e20f4_status.txt 2>/dev/null | wc -l); \
     echo "$f3 $f3a $f4 $f4a $bad $(date +%H:%M)"' 2>/dev/null)
  echo "poll $i: FD003=$out"
  f3=$(echo $out | awk '{print $1}'); f3a=$(echo $out | awk '{print $2}')
  f4=$(echo $out | awk '{print $3}'); f4a=$(echo $out | awk '{print $4}')
  bad=$(echo $out | awk '{print $5}')
  if [ "$bad" != "0" ] && [ -n "$bad" ]; then echo "FAILURES DETECTED: $bad"; fi
  if [ "$f3a" = "1" ] && [ "$f4a" = "1" ]; then echo "BOTH BATCHES COMPLETE"; exit 0; fi
  sleep 300
done
echo "TIMEOUT after 5h"
exit 1
