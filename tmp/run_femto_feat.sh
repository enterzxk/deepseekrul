#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
mkdir -p data/femto_features
nohup $PY scripts/build_femto_features.py --root ~/femto --out data/femto_features   --sets Learning_set Full_Test_Set > logs/femto_feat.log 2>&1 &
echo "PID=$!"
sleep 15
tail -3 logs/femto_feat.log 2>/dev/null || echo starting
