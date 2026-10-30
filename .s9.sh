
cd /home/yndx/zxk/bearing-rul-pi
echo trainers=$(pgrep -c -f 'run_candidate.py')
for f in $(ls -t logs/order03/split9_*.log 2>/dev/null | head -3); do echo "== $(basename $f)"; tail -n 3 "$f" | cut -c1-120; done
