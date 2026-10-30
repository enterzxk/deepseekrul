
cd /home/yndx/zxk/bearing-rul-pi
rm -f /tmp/dsh_split9.lock
setsid nohup bash /home/yndx/p8/split9.sh > /home/yndx/p8/split9c.out 2>&1 < /dev/null &
sleep 25
echo "trainers=$(pgrep -c -f 'run_candidate.py')"
ls logs/order03/split9_*.log 2>/dev/null | wc -l
tail -2 /home/yndx/p8/split9c.out 2>/dev/null
