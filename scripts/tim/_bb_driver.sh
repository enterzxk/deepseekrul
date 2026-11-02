#!/bin/bash
# Driver for the second-backbone run on the A40. Three seeds in parallel, CPU only
# (the GPU is shared and was down to 12 MiB free when this was launched).
set -u
cd "$HOME/bbwork" || exit 1
export RUL_ROOT="$HOME/bbwork"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export BB_VERBOSE=0
PY="$HOME/miniconda3/envs/a40_env/bin/python"
mkdir -p "$HOME/bbwork/logs"

COMMON="--subsets FD001,FD002,FD003,FD004 --folds 0,1,2,3,4 --epochs 30 --patience 5 --stride 4 --device cpu --out $HOME/bbwork/bb"

full () {
  for s in 2027 2028 2029; do
    $PY -u scripts/tim/train_backbone_b.py $COMMON --seeds "$s" > "$HOME/bbwork/logs/s$s.log" 2>&1 &
    echo "launched seed $s (pid $!)"
  done
  wait
  echo "=== all seeds done, building audit ==="
  $PY scripts/tim/train_backbone_b.py --mode audit --out "$HOME/bbwork/bb" 2>&1
  echo "ALL_DONE"
}

case "${1:-full}" in
  full) full ;;
esac
