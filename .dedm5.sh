
pids=$(pgrep -f "[m]5.sh" | sort -n); first=$(echo "$pids" | head -1)
for p in $pids; do [ "$p" != "$first" ] && kill -9 $p 2>/dev/null; done
sleep 2
tp=$(pgrep -f "[r]un_candidate" | sort -n); i=0
for p in $tp; do i=$((i+1)); [ $i -gt 3 ] && kill -9 $p 2>/dev/null; done
sleep 2
echo "controllers=$(pgrep -fc '[m]5.sh') trainers=$(pgrep -c -f '[r]un_candidate')"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
