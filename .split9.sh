#!/bin/bash
exec 8>/tmp/dsh_split9.lock
flock -n 8 || exit 0
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
run_one() {
  t=$1; v=$2; s=$3
  $PY -u scripts/run_candidate.py --subset FD002 --variant $v --seed $s --split-seed 9 \
      --ref-mode global --late-weight 1.8 --epochs 30 --patience 10 --d-model 64 --num-layers 1 \
      --out experiments/split9/$t > logs/order03/split9_${t}_$s.log 2>&1
}
for s in 2027 2028 2029; do run_one M1_global A $s & done
wait
for s in 2027 2028 2029; do run_one M6_tfA tfA $s & done
wait
for s in 2027 2028 2029; do run_one M0_raw_tf tf $s & done
wait
echo "SPLIT9_DONE"
