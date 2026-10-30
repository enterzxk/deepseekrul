
tp=$(pgrep -f "[r]un_candidate" | sort -n); i=0
for p in $tp; do i=$((i+1)); [ $i -gt 3 ] && kill -9 $p 2>/dev/null; done
sleep 3
echo "trainers=$(pgrep -c -f '[r]un_candidate')" 
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
ls /home/yndx/zxk/bearing-rul-pi/experiments/order03/M6_tfA/FD002/*/metrics.json 2>/dev/null | wc -l
