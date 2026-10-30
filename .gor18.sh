
cd /home/yndx/zxk/bearing-rul-pi
if ! pgrep -f "[r]18.sh" > /dev/null; then
  setsid nohup bash /home/yndx/p8/r18.sh > /home/yndx/p8/r18.out 2>&1 < /dev/null &
  sleep 12
fi
echo "runners=$(pgrep -fc '[r]18.sh') trainers=$(pgrep -c -f 'run_candidate.py')"
