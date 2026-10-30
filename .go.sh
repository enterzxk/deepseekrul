
cd /home/yndx/zxk/bearing-rul-pi
if ! pgrep -f "[o]3f.sh" > /dev/null; then
  setsid nohup bash /home/yndx/p8/o3f.sh > /home/yndx/p8/o3f.out 2>&1 < /dev/null &
  sleep 8
fi
echo "controllers=$(pgrep -fc '[o]3f.sh') trainers=$(pgrep -c -f '[r]un_candidate')"
ls experiments/order03/*/*/*/metrics.json | wc -l
