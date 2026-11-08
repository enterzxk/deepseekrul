#!/bin/bash
for p in $(ls /proc | grep -E '^[0-9]+$'); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$c" in *v3b.py*|*q_v3b.sh*) kill -9 $p 2>/dev/null;; esac
done
sleep 1
ps -eo args | grep -c '[v]3b.py'
