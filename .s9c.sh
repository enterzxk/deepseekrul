
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f 'run_candidate.py') m1=$(ls experiments/split9/M1_global/FD002/*/metrics.json 2>/dev/null | wc -l)"
ls -t logs/order03/split9_M1_*.log 2>/dev/null | head -1 | xargs -r tail -n1
