cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f '[r]un_candidate')"
cat /home/yndx/p8/r2.state 2>/dev/null
ls experiments/M3z/FD002/*/metrics.json 2>/dev/null | wc -l
tail -n 1 $(ls -t logs/order03/M3z_*.log 2>/dev/null | head -1) 2>/dev/null | cut -c1-80