
for p in $(pgrep -f "[r]un_candidate"); do ps -o stat= -p $p; done | sort | uniq -c
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
ls /home/yndx/zxk/bearing-rul-pi/experiments/order03/*/*/*/metrics.json 2>/dev/null | wc -l
