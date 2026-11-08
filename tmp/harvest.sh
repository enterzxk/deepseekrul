#!/bin/bash
cd ~/deepseek-rul
echo "point files: $(ls experiments/punet_e2e/*/f*_s42/e2e_point_none.json 2>/dev/null | wc -l)"
for S in FD001 FD002 FD003 FD004; do
  n=$(ls experiments/punet_e2e/$S/f*_s42/e2e_point_none.json 2>/dev/null | wc -l)
  echo "  $S: $n/5"
done
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
/home/yndx/miniconda3/envs/a40_env/bin/python scripts/make_tables.py 2>&1 | tail -30
