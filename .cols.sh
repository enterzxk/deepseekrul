#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
head -1 experiments/cplab2/FD002/A_s2027/matched_cutoff_predictions.csv
echo "---"
head -1 experiments/order03/M1_global/FD002/A_s2027/matched_cutoff_predictions.csv
echo "---"
wc -l experiments/cplab2/FD002/A_s2027/matched_cutoff_predictions.csv
