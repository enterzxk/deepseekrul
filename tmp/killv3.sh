#!/bin/bash
for p in $(pgrep -f 'q_v3a.sh|run_v3.py'); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$c" in *q_v3a.sh*|*run_v3.py*) echo "kill $p"; kill -9 $p 2>/dev/null;; esac
done
sleep 2
ps -eo args | grep -E '[q]_v3a|[r]un_v3' || echo none
