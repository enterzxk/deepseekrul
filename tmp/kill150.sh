#!/bin/bash
for p in $(pgrep -f run_full_fold3); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$c" in
    *w150*) echo "kill $p"; kill $p;;
  esac
done
sleep 2
ps -eo pid,etime,args | grep '[r]un_full_fold3' || true
