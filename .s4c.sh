
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f 'run_candidate.py') done=$(ls experiments/shuf/A/FD002/*/metrics.json 2>/dev/null | wc -l)"
cat /home/yndx/p8/shuf4.state 2>/dev/null
ls -t logs/order03/shuf4_*.log 2>/dev/null | head -1 | xargs -r tail -n 1
