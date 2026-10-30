
cd /home/yndx/zxk/bearing-rul-pi
echo "--- running cells (recent logs) ---"
for f in $(ls -t logs/order03/*.log | head -6); do echo -n "$(basename $f): "; tail -n 1 "$f" | cut -c1-100; done
tp=$(pgrep -f "[r]un_candidate" | sort -n); i=0
for p in $tp; do i=$((i+1)); [ $i -gt 3 ] && kill -9 $p 2>/dev/null; done
sleep 2
echo "trainers=$(pgrep -c -f '[r]un_candidate') supervisors=$(pgrep -fc '[s]up.sh')"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
