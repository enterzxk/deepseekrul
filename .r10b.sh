
cd /home/yndx/zxk/bearing-rul-pi
echo "--- recent logs ---"
for f in $(ls -t logs/order03/*.log | head -8); do echo -n "$(basename $f) | "; tail -n 1 "$f" | cut -c1-110; echo; done
echo "--- supervisor out ---"
tail -n 5 /home/yndx/p8/sup3.out 2>/dev/null
echo "--- queue ---"
cat /home/yndx/p8/queue.txt
