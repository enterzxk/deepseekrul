
cd /home/yndx/zxk/bearing-rul-pi
cp /home/yndx/p8/queue2.txt /home/yndx/p8/queue.txt 2>/dev/null
if ! pgrep -f "[s]up.sh" > /dev/null; then
  setsid nohup bash /home/yndx/p8/sup.sh /home/yndx/p8/queue.txt > /home/yndx/p8/sup2.out 2>&1 < /dev/null &
  sleep 10
fi
echo "supervisors=$(pgrep -fc '[s]up.sh') trainers=$(pgrep -c -f '[r]un_candidate')"
echo "cells=$(ls experiments/order03/*/*/*/metrics.json | wc -l)"
