
pkill -9 -f "[r]un_candidate" 2>/dev/null
sleep 6
pkill -9 -f "[r]un_candidate" 2>/dev/null
sleep 4
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "procs: $(pgrep -c -f '[r]un_candidate')"
cd /home/yndx/zxk/bearing-rul-pi
setsid nohup bash /home/yndx/p8/o3.sh > /home/yndx/p8/o3.out 2>&1 < /dev/null &
sleep 10
echo "after launch procs: $(pgrep -c -f '[r]un_candidate') scripts: $(pgrep -fc '[o]3.sh')"
