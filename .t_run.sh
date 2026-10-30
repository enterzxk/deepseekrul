#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
mkdir -p logs/tournament
run_one() {
  $PY -u scripts/run_candidate.py --subset $1 --variant $2 --seed $3 --epochs 40 --patience 15 --d-model 88       --out experiments/tournament > logs/tournament/$1_$2_$3.log 2>&1
}
export -f run_one
export PY
for ds in FD001 FD002; do
  for v in base A AB ABE ABC ABCE; do
    for s in 2027 2028 2029; do
      echo "$ds $v $s"
    done
  done
done | xargs -P 6 -n 3 bash -c 'run_one "$@"' _
echo TOURNAMENT_DONE
