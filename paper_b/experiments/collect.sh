#!/bin/bash
# Run on the machine that holds the results (lab3090), then copy the summaries.
set -e
cd /home/user/rulb
PY=~/miniconda3/envs/mathm/bin/python
$PY -m paper_b.experiments.analyze --root paper_b/results/raw --out paper_b/results/all_runs.csv
$PY -m paper_b.experiments.make_tables --results paper_b/results --out paper_b/results/tables
echo "collected"
