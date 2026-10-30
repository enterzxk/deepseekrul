
cd /home/yndx/zxk/bearing-rul-pi
if ! pgrep -f "[m]7final.sh" > /dev/null && [ "$(ls experiments/order03/M7_rawback/FD002/*/metrics.json 2>/dev/null | wc -l)" -lt 3 ]; then
  setsid nohup bash /home/yndx/p8/m7final.sh > /home/yndx/p8/m7final.out 2>&1 < /dev/null &
  sleep 10
fi
echo "runners=$(pgrep -fc '[m]7final.sh') trainers=$(pgrep -c -f '[r]un_candidate')"
