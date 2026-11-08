#!/bin/bash
pkill -9 -f run_femto_cqr 2>/dev/null
sleep 2
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
unset CUDA_VISIBLE_DEVICES
rm -rf experiments/femto
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
nohup $PY scripts/run_femto_cqr.py --feat-dir data/femto_features --window 30 --epochs 30   --max-windows 300 --label rel   --only Bearing1_1,Bearing1_2,Bearing2_1,Bearing2_2,Bearing3_1,Bearing3_2   --out experiments/femto > logs/femto_rel.log 2>&1 &
echo "PID=$!"
sleep 45
head -2 logs/femto_rel.log
