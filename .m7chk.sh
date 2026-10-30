
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f 'run_candidate.py')"
tail -n 1 $(ls -t logs/order03/M7v2_*.log | head -1) | cut -c1-70
ls experiments/order03/M7_rawback/FD002/*/metrics.json 2>/dev/null | wc -l
