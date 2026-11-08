#!/bin/bash
# keep one worker per subset (the one handling folds 0-2), kill the "3 4" workers
for pid in $(ps -eo pid,cmd | grep '[r]un_matrix8.sh' | awk '{print $1}'); do :; done
# kill python children whose parent bash was started with "3 4"
for pid in $(ps -eo pid,ppid,cmd | awk '/run_e2e_cqr.py/ && !/awk/ {print $1}'); do
  parent=$(ps -o ppid= -p $pid 2>/dev/null | tr -d ' ')
  pcmd=$(ps -o cmd= -p $parent 2>/dev/null)
  case "$pcmd" in
    *"3 4"*) kill -9 $pid 2>/dev/null; echo "killed $pid (fold 3-4 worker)";;
  esac
done
sleep 3
echo "python procs now: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
