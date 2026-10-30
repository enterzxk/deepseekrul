#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi/data/CMAPSS || exit 1
echo 'file                 perms        readable-by-me'
for f in test_FD001.txt test_FD002.txt test_FD003.txt test_FD004.txt RUL_FD001.txt RUL_FD002.txt RUL_FD003.txt RUL_FD004.txt; do
  if [ -e "$f" ]; then
    p=$(stat -c '%A %U:%G' "$f" 2>/dev/null)
    if [ -r "$f" ]; then r=YES; else r=NO; fi
    printf '%-20s %-12s %s\n' "$f" "$p" "$r"
  else
    printf '%-20s %-12s %s\n' "$f" 'MISSING' '-'
  fi
done
