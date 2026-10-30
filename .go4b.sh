
pkill -9 -f "[r]un_candidate" 2>/dev/null
sleep 3
cd /home/yndx/zxk/bearing-rul-pi
setsid nohup bash /home/yndx/p8/r4q.sh > /home/yndx/p8/r4q2.out 2>&1 < /dev/null &
sleep 12
echo "controllers=$(pgrep -fc '[r]4q.sh') trainers=$(pgrep -c -f '[r]un_candidate')"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
