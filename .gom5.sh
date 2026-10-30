
cd /home/yndx/zxk/bearing-rul-pi
if ! pgrep -f "[m]5.sh" > /dev/null; then
  setsid nohup bash /home/yndx/p8/m5.sh > /home/yndx/p8/m5.out 2>&1 < /dev/null &
  sleep 10
fi
echo "controllers=$(pgrep -fc '[m]5.sh') trainers=$(pgrep -c -f '[r]un_candidate')"
