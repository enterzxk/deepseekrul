
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f 'run_candidate.py') done=$(ls experiments/split8/*/FD002/*/metrics.json 2>/dev/null | wc -l)"
ls -t logs/order03/split8_*.log 2>/dev/null | head -2 | xargs -r tail -n 1
