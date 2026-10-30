
cd /home/yndx/zxk/bearing-rul-pi
if ! pgrep -f "[b]atch.sh" > /dev/null; then
  setsid nohup bash /home/yndx/p8/batch.sh > /home/yndx/p8/batch.out 2>&1 < /dev/null &
  sleep 12
fi
echo "runners=$(pgrep -fc '[b]atch.sh') trainers=$(pgrep -c -f '[r]un_candidate')"
