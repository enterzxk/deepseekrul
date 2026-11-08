#!/bin/bash
for p in $(ls /proc | grep -E '^[0-9]+$'); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$c" in *'--tag A --l1'*|*'--tag AB '*|*q_v3a.sh*) kill -9 $p 2>/dev/null;; esac
done
sleep 1
ps -eo args | grep -c '[r]un_v3.py'
