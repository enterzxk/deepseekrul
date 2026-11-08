#!/bin/bash
for p in $(pgrep -f run_full_fold3); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$c" in *tag\ big*) echo "kill $p"; kill $p;; esac
done
sleep 2
ps -eo args | grep '[r]un_full_fold3' || echo none
