#!/bin/bash
pkill -9 -f 'r3cycle_detached' 2>/dev/null; echo k1=$?
pkill -9 -f 'r3cycle.sh' 2>/dev/null; echo k2=$?
pkill -9 -f 'run_candidate.py' 2>/dev/null; echo k3=$?
sleep 4
echo 'remaining:'; pgrep -fa 'run_candidate|r3cycle' | head -10
echo 'count:'; pgrep -c -f 'run_candidate|r3cycle' || echo 0
rm -rf /home/yndx/zxk/bearing-rul-pi/experiments/r3_cycle
echo 'dir:'; ls /home/yndx/zxk/bearing-rul-pi/experiments/ | grep r3_cycle || echo none
