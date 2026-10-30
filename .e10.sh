#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PYBIN=/home/yndx/miniconda3/envs/a40_env/bin/python
SHA=aa7072f745c9ea2f552fb7e3203d1b14ef1501f62d01e445a919329adc1453b4
rm -f /tmp/e10_status.txt /tmp/e10_jobs.txt
for fold in 1 2 3 4; do
  for arm in A PE; do
    for seed in 2027 2028 2029; do
      echo "$arm $seed $fold" >> /tmp/e10_jobs.txt
    done
  done
done
echo "queued $(wc -l < /tmp/e10_jobs.txt) runs at $(date -Is)" >> /tmp/e10_status.txt
cat /tmp/e10_jobs.txt | xargs -P 3 -n 3 bash -c '
  arm=$0; seed=$1; fold=$2
  out=experiments/e10/${arm}_s${seed}_f${fold}
  mkdir -p $out
  /home/yndx/miniconda3/envs/a40_env/bin/python -u scripts/run_candidate.py \
    --subset FD002 --variant $arm --seed $seed --split-seed 7 --fold $fold \
    --ref-mode global --epochs 80 --patience 20 --d-model 64 --num-layers 1 \
    --prereg-sha '"$SHA"' --out $out > /tmp/e10_${arm}_${seed}_f${fold}.log 2>&1
  echo "done $arm $seed f$fold rc=$? at $(date -Is)" >> /tmp/e10_status.txt
'
echo "ALLDONE $(date -Is)" >> /tmp/e10_status.txt
