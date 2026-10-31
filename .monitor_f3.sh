#!/bin/bash
# Poll FD003 on lab-a40 every 3 minutes until ALLDONE (max ~1.5h).
for i in $(seq 1 30); do
  out=$(timeout 60 ssh lab-a40 -o BatchMode=yes -o ConnectTimeout=20 -o UpdateHostKeys=no \
    'd=$(grep -c "^done FD003" /tmp/e20f3_status.txt 2>/dev/null); a=$(grep -c ALLDONE /tmp/e20f3_status.txt 2>/dev/null); \
     bad=$(grep -c "rc=[^0]" /tmp/e20f3_status.txt 2>/dev/null); echo "$d $a $bad $(date +%H:%M)"' 2>/dev/null)
  echo "poll $i: $out"
  d=$(echo $out | awk '{print $1}'); a=$(echo $out | awk '{print $2}')
  if [ "$a" = "1" ]; then echo "FD003 COMPLETE ($d/44)"; exit 0; fi
  sleep 180
done
echo "TIMEOUT"; exit 1
