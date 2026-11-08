#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
nohup $PY scripts/run_punet_stage.py --subset FD001 --fold 0 --seed 42 --stage all   > logs/punet_fd001.log 2>&1 &
echo "PID=$!"
sleep 40
tail -8 logs/punet_fd001.log
