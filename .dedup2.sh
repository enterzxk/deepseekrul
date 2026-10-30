
cd /home/yndx/zxk/bearing-rul-pi
echo "--- controllers ---"
pgrep -af "[o]3f.sh|[o]3d.sh|[h]2.sh" | head
# keep only the most recently started controller (h2.sh is the current batch)
for p in $(pgrep -f "[o]3f.sh"); do kill -9 $p 2>/dev/null; done
for p in $(pgrep -f "[o]3d.sh"); do kill -9 $p 2>/dev/null; done
sleep 2
echo "controllers after: $(pgrep -fc '[h]2.sh')"
# keep 3 trainers, kill the rest (newest first)
pids=$(pgrep -f "[r]un_candidate" | sort -n)
count=$(echo "$pids" | wc -l)
echo "trainers before: $count"
i=0
for p in $pids; do
  i=$((i+1))
  if [ $i -gt 3 ]; then kill -9 $p 2>/dev/null; fi
done
sleep 3
echo "trainers after: $(pgrep -c -f '[r]un_candidate')"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
