#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:/
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
rm -f /tmp/t1t2_status.txt
for seed in 2027 2028 2029; do
  out=experiments/e3fix/BEfix_s${seed}
  mkdir -p $out
  nohup $PY -u scripts/run_candidate.py --subset FD002 --variant BE --seed $seed     --split-seed 7 --ref-mode global --epochs 80 --patience 80 --d-model 64 --num-layers 1     --prereg-sha c8f6d653a8a08fbd25eda09bf318300c0301a05977c06de878cc796b0051c8d6 --out $out > /tmp/t1_BEfix_${seed}.log 2>&1 &
done
echo "T1 launched" >> /tmp/t1t2_status.txt
wait
echo "T1 done" >> /tmp/t1t2_status.txt
# T2: clean dir
rm -rf /home/yndx/zxk/e4_clean
mkdir -p /home/yndx/zxk/e4_clean
cp -r /home/yndx/zxk/bearing-rul-pi/src /home/yndx/zxk/bearing-rul-pi/scripts /home/yndx/zxk/e4_clean/
ln -s /home/yndx/zxk/bearing-rul-pi/data /home/yndx/zxk/e4_clean/data
cd /home/yndx/zxk/e4_clean || exit 1
$PY -u scripts/run_candidate.py --subset FD002 --variant A --seed 2027   --split-seed 7 --ref-mode global --epochs 80 --patience 20 --d-model 64 --num-layers 1   --prereg-sha dc83ea8be781f18dd54419cf291638ee5d8445736440a2aa8a6ff43d233c4ede --out experiments/e4 > /tmp/t2_Afresh.log 2>&1
echo "T2 done" >> /tmp/t1t2_status.txt
