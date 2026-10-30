
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f 'run_candidate.py')"
cat /home/yndx/p8/shuf4.state 2>/dev/null
ls -t logs/order03/shuf4_*.log 2>/dev/null | head -1 | xargs -r tail -n 2
