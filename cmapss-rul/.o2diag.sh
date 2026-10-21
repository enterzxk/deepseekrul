
cd /home/yndx/zxk/bearing-rul-pi
ls -t logs/order02b/ | head -3
for f in $(ls -t logs/order02b/*.log 2>/dev/null | head -2); do echo "== $f"; tail -n 4 "$f"; done
pgrep -c -f run_candidate
