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
  [ -f "$f" ] && { echo "skip $tag $ds $v $s"; return 0; }
  $PY -u scripts/run_candidate.py --subset $ds --variant $v --seed $s --split-seed 7 \
      --ref-mode $ref --late-weight 1.8 --epochs 30 --patience 10 --d-model 64 --num-layers 1 \
      --out $out > logs/order03/${tag}_${ds}_${s}.log 2>&1
}
export -f run_cell
export PY PYTORCH_CUDA_ALLOC_CONF
{
  for s in 2027 2028 2029; do
    echo "M0_raw_tf FD002 tf cond $s"
    echo "M1_global FD002 A global $s"
    echo "M2_pop FD002 A pop $s"
    echo "M3_cond FD002 A cond $s"
    echo "M4_racdr FD002 RACDR cond $s"
    echo "M0_raw_tf FD001 tf cond $s"
    echo "M3_cond FD001 A cond $s"
    echo "M4_racdr FD001 RACDR cond $s"
  done
} | xargs -P 3 -n 5 bash -c 'run_cell "$@"' _
echo ORDER03_DONE
