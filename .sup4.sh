
cd /home/yndx/zxk/bearing-rul-pi
if ! pgrep -f "[s]up.sh" > /dev/null; then
  setsid nohup bash /home/yndx/p8/sup.sh /home/yndx/p8/queue.txt > /home/yndx/p8/sup4.out 2>&1 < /dev/null &
  sleep 12
fi
echo "supervisors=$(pgrep -fc '[s]up.sh') trainers=$(pgrep -c -f '[r]un_candidate')"
