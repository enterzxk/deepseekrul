
cd /home/yndx/zxk/bearing-rul-pi
echo "--- out ---"; cat /home/yndx/p8/split8.out 2>/dev/null | tail -5
echo "--- manual launch test ---"
setsid nohup bash /home/yndx/p8/split8.sh > /home/yndx/p8/split8b.out 2>&1 < /dev/null &
sleep 20
echo "runners=$(pgrep -fc '[s]plit8.sh') trainers=$(pgrep -c -f 'run_candidate.py')"
tail -3 /home/yndx/p8/split8b.out 2>/dev/null
ls -t logs/order03/split8_*.log 2>/dev/null | head -1 | xargs -r tail -n 2
