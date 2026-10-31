#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: > /tmp/e20_status.txt
rm -f /tmp/e20_jobs.txt
for sub in FD003 FD004; do
  for arm in A BE PE; do
    for s in 2027 2028 2029; do
      for f in 0 1 2 3 4; do echo "$sub $arm $s $f" >> /tmp/e20_jobs.txt; done
    done
  done
done
echo "queued $(wc -l < /tmp/e20_jobs.txt) at $(date -Is)" >> /tmp/e20_status.txt
# FD003 gets 2 slots, FD004 gets 4 (it is the bottleneck)
grep "^FD003" /tmp/e20_jobs.txt > /tmp/e20_fd3.txt
grep "^FD004" /tmp/e20_jobs.txt > /tmp/e20_fd4.txt
run_one () {
  sub=$1; arm=$2; seed=$3; fold=$4
  extra=""
  [ "$sub" = "FD004" ] && extra="--num-layers 1"
  /home/yndx/miniconda3/envs/a40_env/bin/python -u scripts/run_candidate.py \
    --subset $sub --variant $arm --seed $seed --split-seed 7 --fold $fold \
    --ref-mode global --epochs 80 --patience 20 --d-model 64 $extra \
    --prereg-sha 1959e9244ea8d14f9bdc71152e793e986604b489cd43352285e3c113fefe1feb --out experiments/e20/$sub/${arm}_s${seed}_f${fold} \
    > /tmp/e20_${sub}_${arm}_${seed}_f${fold}.log 2>&1
  echo "done $sub $arm $seed f$fold rc=$? $(date +%H:%M)" >> /tmp/e20_status.txt
}
export -f run_one
xargs -P 2 -a /tmp/e20_fd3.txt bash -c 'run_one "$@"' _
xargs -P 4 -a /tmp/e20_fd4.txt bash -c 'run_one "$@"' _
echo ALLDONE >> /tmp/e20_status.txt
