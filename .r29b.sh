
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f 'run_candidate.py') done=$(ls experiments/shuf/A/FD002/*/metrics.json 2>/dev/null | wc -l)"
ls -t logs/order03/shuf2_*.log 2>/dev/null | head -1 | xargs -r tail -n 1
