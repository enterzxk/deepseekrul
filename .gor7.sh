
cd /home/yndx/zxk/bearing-rul-pi
if ! pgrep -f "[r]7q.sh" > /dev/null; then
  setsid nohup bash /home/yndx/p8/r7q.sh > /home/yndx/p8/r7q.out 2>&1 < /dev/null &
  sleep 12
fi
echo "controllers=$(pgrep -fc '[r]7q.sh') trainers=$(pgrep -c -f '[r]un_candidate')"
