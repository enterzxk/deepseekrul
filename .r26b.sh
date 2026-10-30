
cd /home/yndx/zxk/bearing-rul-pi
echo "done=$(ls experiments/split9/M1_global/FD002/*/metrics.json 2>/dev/null | wc -l) trainers=$(pgrep -c -f 'run_candidate.py')"
ls -t logs/order03/split9_M1_*.log 2>/dev/null | head -1 | xargs -r tail -n1
