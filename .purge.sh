
cd /home/yndx/zxk/bearing-rul-pi
for i in 1 2 3; do
  pkill -9 -f "[b]atch.sh" 2>/dev/null
  pkill -9 -f "[s]up.sh" 2>/dev/null
  pkill -9 -f "run_candidate.py" 2>/dev/null
  sleep 3
done
echo "after purge: trainers=$(pgrep -c -f 'run_candidate.py') scripts=$(pgrep -fc '[b]atch.sh')"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
