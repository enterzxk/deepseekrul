#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for s in dsh_var_balanced dsh_split_census dsh_pooled_compare dsh_prediction_corr dsh_conformal; do
  if [ -f scripts/$s.py ]; then
    out=$($PY scripts/$s.py 2>&1 | tail -n 2 | tr '\n' ' ' | cut -c1-90)
    echo "OK   $s -> $out"
  else
    echo "MISS $s"
  fi
done
