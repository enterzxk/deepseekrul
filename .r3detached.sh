#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
cat > /tmp/r3cycle_detached.sh <<'EOS'
#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
for var in A BE; do
  for seed in 2027 2028 2029; do
    out=experiments/r3_cycle/${var}_s${seed}
    if [ -f $out/metrics.json ]; then echo "skip $out" >> /tmp/r3cycle_status.txt; continue; fi
    $PY -u scripts/run_candidate.py --subset FD002 --variant $var --seed $seed --split-seed 7 \
      --ref-mode global --epochs 30 --patience 30 --d-model 64 --num-layers 1 \
      --out $out > /tmp/r3c_${var}_${seed}.log 2>&1
    echo "$var s$seed exit=$? $(tail -n1 /tmp/r3c_${var}_${seed}.log)" >> /tmp/r3cycle_status.txt
  done
done
echo ALLDONE >> /tmp/r3cycle_status.txt
EOS
chmod +x /tmp/r3cycle_detached.sh
rm -f /tmp/r3cycle_status.txt
nohup /tmp/r3cycle_detached.sh >/dev/null 2>&1 &
echo started pid=$!
