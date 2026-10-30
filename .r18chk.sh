
cd /home/yndx/zxk/bearing-rul-pi
cat /home/yndx/p8/r18.out 2>/dev/null | tail -5
echo "--- logs ---"
ls -t logs/order03/H2_fixed1_FD001_*.log 2>/dev/null | head -1 | xargs -r tail -n 2
ls -t logs/order03/M4b_popgate_FD002_203*.log 2>/dev/null | head -1 | xargs -r tail -n 2
echo "trainers=$(pgrep -c -f 'run_candidate.py')"
