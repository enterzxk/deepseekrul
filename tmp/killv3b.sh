#!/bin/bash
for round in 1 2 3; do
  found=0
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$c" in
      *run_v3.py*|*q_v3a.sh*|*tasks_v3a*) echo "kill $p $c"; kill -9 $p 2>/dev/null; found=1;;
    esac
  done
  [ $found -eq 0 ] && break
  sleep 2
done
sleep 2
ps -eo args | grep -E '[r]un_v3|[q]_v3a' || echo CLEAN
