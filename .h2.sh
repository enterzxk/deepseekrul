#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
mkdir -p logs/order03
run_cell() {
  tag=$1; ds=$2; v=$3; ref=$4; s=$5
  out=experiments/order03/$tag
  f="$out/$ds/${v}_s${s}/metrics.json"
  [ -f "$f" ] && return 0
  $PY -u scripts/run_candidate.py --subset $ds --variant $v --seed $s --split-seed 7 \
      --ref-mode $ref --late-weight 1.8 --epochs 30 --patience 10 --d-model 64 --num-layers 1 \
      --out $out > logs/order03/${tag}_${ds}_${s}.log 2>&1
}
export -f run_cell
export PY PYTORCH_CUDA_ALLOC_CONF
{
  for s in 2027 2028 2029; do
    echo "M4_ckpt FD002 RACDR cond $s"
    echo "H2_fixed1 FD002 RACDR_fixed1 cond $s"
    echo "H2_const FD002 RACDR_const cond $s"
    echo "H2_half FD002 RACDR_half cond $s"
  done
} | xargs -P 3 -n 5 bash -c 'run_cell "$@"' _
echo H2_DONE
