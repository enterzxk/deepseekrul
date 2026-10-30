
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f 'run_candidate.py')"
ls experiments/split8/*/FD002/*/metrics.json 2>/dev/null | sed 's|experiments/split8/||'
