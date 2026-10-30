#!/bin/bash
exec 8>/tmp/dsh_fd001.lock
flock -n 8 || exit 0
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
pids=""
run_one() {
  t=$1; d=$2; v=$3; s=$4
  f="experiments/order03/$t/$d/${v}_s$s/metrics.json"
  [ -f "$f" ] && return 0
  $PY -u scripts/run_candidate.py --subset $d --variant $v --seed $s --split-seed 7 \
      --ref-mode global --late-weight 1.8 --epochs 30 --patience 10 --d-model 64 --num-layers 1 \
      --out experiments/order03/$t > logs/order03/${t}_${d}_$s.log 2>&1
}
export -f run_one
for s in 2027 2028 2029; do
  run_one M6_tfA FD001 tfA $s &
  run_one M1_global FD001 A $s &
done
wait
echo "FD001_DONE"
