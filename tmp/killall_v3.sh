#!/bin/bash
for round in 1 2 3; do
  n=0
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$c" in
      *v3a.sh*|*v3b.sh*|*'run_v3'*|*xargs*tasks_v3*) kill -9 $p 2>/dev/null; n=$((n+1));;
    esac
  done
  [ $n -eq 0 ] && break
  sleep 2
done
sleep 1
ps -eo args | grep -c '[r]un_v3'
