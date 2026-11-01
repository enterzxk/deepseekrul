#!/bin/bash
# CET-RUL runner on the 3090 box. Usage: ./run_cet.sh SUBSET FOLD SEED EPOCHS [ABLATION]
set -e
cd /home/user/yr/zxk/bearing-rul-pi
export RUL_ROOT=/home/user/yr/zxk/bearing-rul-pi
export PYTHONPATH=$RUL_ROOT/src:$RUL_ROOT/scripts
PY=/home/user/miniconda3/envs/baseline/bin/python
SUB=${1:-FD001}; FOLD=${2:-0}; SEED=${3:-2027}; EP=${4:-40}; ABL=${5:-full}
TAG=""; [ "$ABL" != "full" ] && TAG="_$ABL"
OUT=experiments/cet/${SUB}/f${FOLD}_s${SEED}${TAG}
mkdir -p "$OUT" logs
$PY scripts/cet_train.py --subset $SUB --fold $FOLD --seed $SEED --epochs $EP \
    --batch 12 --ablation "$ABL" --out "$OUT" 2>&1 | tee "logs/cet_${SUB}_f${FOLD}_s${SEED}${TAG}.log"
