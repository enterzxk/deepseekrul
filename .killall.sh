
pkill -9 -f "[x]args -P" 2>/dev/null
pkill -9 -f "[r]un_cell" 2>/dev/null
pkill -9 -f "[h]2.sh" 2>/dev/null
pkill -9 -f "[m]1.sh" 2>/dev/null
pkill -9 -f "[o]3f.sh" 2>/dev/null
pkill -9 -f "[o]3d.sh" 2>/dev/null
sleep 4
pkill -9 -f "[r]un_candidate" 2>/dev/null
sleep 3
echo "after kill: trainers=$(pgrep -c -f '[r]un_candidate') controllers=$(pgrep -fc '[h]2.sh|[m]1.sh|[o]3')"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
