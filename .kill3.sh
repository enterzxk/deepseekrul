
pkill -9 -f "[x]args -P 3" 2>/dev/null
pkill -9 -f "[r]un_cell" 2>/dev/null
pkill -9 -f "[r]un_candidate" 2>/dev/null
sleep 5
pkill -9 -f "[r]un_candidate" 2>/dev/null
sleep 3
echo "after kill: $(pgrep -c -f '[r]un_candidate')"
cd /home/yndx/zxk/bearing-rul-pi
setsid nohup bash /home/yndx/p8/o3.sh > /home/yndx/p8/o3.out 2>&1 < /dev/null &
sleep 12
echo "after launch: $(pgrep -c -f '[r]un_candidate') xargs: $(pgrep -fc '[x]args -P 3')"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
