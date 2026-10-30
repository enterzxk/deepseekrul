#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
grep -n '_sha\|^def main' scripts/run_candidate.py | head
