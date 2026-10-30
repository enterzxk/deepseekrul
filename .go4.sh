
cd /home/yndx/zxk/bearing-rul-pi
if ! pgrep -f "[r]4q.sh" > /dev/null; then
  setsid nohup bash /home/yndx/p8/r4q.sh > /home/yndx/p8/r4q.out 2>&1 < /dev/null &
  sleep 10
fi
echo "controllers=$(pgrep -fc '[r]4q.sh') trainers=$(pgrep -c -f '[r]un_candidate')"
ls experiments/order03/M1_global/*/*/metrics.json 2>/dev/null | wc -l
