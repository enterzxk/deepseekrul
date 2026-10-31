#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: > /tmp/e20fix_status.txt
rm -f /tmp/e20fix_fd3.txt /tmp/e20fix_fd4.txt
for arm in A BE PE; do
  for s in 2027 2028 2029; do
    for f in 0 1 2 3 4; do
      # FD003: skip the one run that already succeeded
      [ "$arm" = "A" ] && [ "$s" = "2027" ] && [ "$f" = "0" ] && continue
      [ ! -f experiments/e20/FD003/${arm}_s${s}_f${f}/FD003/${arm}_s${s}/metrics.json ] && \
        echo "FD003 $arm $s $f" >> /tmp/e20fix_fd3.txt
      # FD004: folds 0-3 are complete; only fold 4 is missing
      [ "$f" = "4" ] && [ ! -f experiments/e20/FD004/${arm}_s${s}_f${f}/FD004/${arm}_s${s}/metrics.json ] && \
        echo "FD004 $arm $s $f" >> /tmp/e20fix_fd4.txt
    done
  done
done
echo "FD003 todo $(wc -l < /tmp/e20fix_fd3.txt), FD004f4 todo $(wc -l < /tmp/e20fix_fd4.txt) at $(date -Is)" >> /tmp/e20fix_status.txt
run_one () {
  sub=$1; arm=$2; seed=$3; fold=$4
  /home/yndx/miniconda3/envs/a40_env/bin/python -u scripts/run_candidate.py \
    --subset $sub --variant $arm --seed $seed --split-seed 7 --fold $fold \
    --ref-mode global --epochs 80 --patience 20 --d-model 64 --num-layers 1 \
    --prereg-sha 1959e9244ea8d14f9bdc71152e793e986604b489cd43352285e3c113fefe1feb --out experiments/e20/$sub/${arm}_s${seed}_f${fold} \
    > /tmp/e20_${sub}_${arm}_${seed}_f${fold}.log 2>&1
  echo "done $sub $arm $seed f$fold rc=$? $(date +%H:%M)" >> /tmp/e20fix_status.txt
}
export -f run_one
xargs -P 4 -n 3 -a /tmp/e20fix_fd3.txt bash -c 'run_one "$@"' _ &
xargs -P 2 -n 3 -a /tmp/e20fix_fd4.txt bash -c 'run_one "$@"' _ &
wait
echo ALLDONE >> /tmp/e20fix_status.txt
