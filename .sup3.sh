
cd /home/yndx/zxk/bearing-rul-pi
setsid nohup bash /home/yndx/p8/sup.sh /home/yndx/p8/queue.txt > /home/yndx/p8/sup3.out 2>&1 < /dev/null &
sleep 15
echo "supervisors=$(pgrep -fc '[s]up.sh') trainers=$(pgrep -c -f '[r]un_candidate')"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
tail -n 5 /home/yndx/p8/sup3.out
