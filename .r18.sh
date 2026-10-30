#!/bin/bash
exec 8>/tmp/dsh_r18.lock
flock -n 8 || exit 0
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
run_one() {
  t=$1; d=$2; v=$3; r=$4; s=$5
  f="experiments/order03/$t/$d/${v}_s$s/metrics.json"
  [ -f "$f" ] && return 0
  $PY -u scripts/run_candidate.py --subset $d --variant $v --seed $s --split-seed 7 \
      --ref-mode $r --late-weight 1.8 --epochs 30 --patience 10 --d-model 64 --num-layers 1 \
      --out experiments/order03/$t > logs/order03/${t}_${d}_$s.log 2>&1
}
export -f run_one
run_one M4b_popgate FD002 RACDR pop 2030 &
run_one M4b_popgate FD002 RACDR pop 2031 &
run_one H2_fixed1 FD002 RACDR_fixed1 pop 2030 &
wait
run_one H2_fixed1 FD001 RACDR_fixed1 pop 2027 &
run_one H2_fixed1 FD001 RACDR_fixed1 pop 2028 &
run_one H2_fixed1 FD001 RACDR_fixed1 pop 2029 &
wait
run_one H2_fixed1 FD002 RACDR_fixed1 pop 2031 &
run_one M4b_popgate FD001 RACDR pop 2030 &
run_one M4b_popgate FD001 RACDR pop 2031 &
wait
echo "R18_DONE"
