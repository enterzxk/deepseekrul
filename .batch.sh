#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
exec 8>/tmp/dsh_batch.lock
flock -n 8 || { echo 'batch already running'; exit 0; }
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
run_batch() {
  t=$1; d=$2; v=$3; r=$4; shift 4
  for s in "$@"; do
    f="experiments/order03/$t/$d/${v}_s$s/metrics.json"
    [ -f "$f" ] && continue
    $PY -u scripts/run_candidate.py --subset $d --variant $v --seed $s --split-seed 7 \
        --ref-mode $r --late-weight 1.8 --epochs 30 --patience 10 --d-model 64 --num-layers 1 \
        --out experiments/order03/$t > logs/order03/${t}_${d}_$s.log 2>&1 &
  done
  wait
}
run_batch M7_rawback FD002 base global 2027 2028 2029
run_batch M7_rawback FD001 base global 2027 2028 2029
run_batch M6_tfA FD001 tfA global 2027 2028 2029
run_batch M1_global FD001 A global 2028 2029
echo "BATCH_ALL_DONE $(date)"
