#!/bin/bash
pkill -9 -f run_femto_cqr 2>/dev/null
sleep 2
echo "femto procs: $(ps -eo cmd | grep -c '[r]un_femto_cqr')"
echo "matrix procs: $(ps -eo cmd | grep -c '[r]un_e2e_cqr.py')"
echo "cells done: $(ls ~/deepseek-rul/experiments/punet_matrix/*/f*_s*/preds_cqr.csv 2>/dev/null | wc -l)"
