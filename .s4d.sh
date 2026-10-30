
cd /home/yndx/zxk/bearing-rul-pi
for f in logs/order03/shuf4_*.log; do echo "== $f"; head -3 "$f"; echo "..."; tail -5 "$f"; done 2>/dev/null | head -40
