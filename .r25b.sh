
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f 'run_candidate.py') done=$(ls experiments/split9/*/FD002/*/metrics.json 2>/dev/null | wc -l)"
ls experiments/split9/*/FD002/*/metrics.json 2>/dev/null | sed 's|experiments/split9/||'
ls -t logs/order03/split9_*.log 2>/dev/null | head -1 | xargs -r tail -n1
