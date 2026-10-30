
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f '[r]un_candidate')"
echo "M7=$(ls experiments/order03/M7_rawback/*/*/metrics.json 2>/dev/null | wc -l) M6FD001=$(ls experiments/order03/M6_tfA/FD001/*/metrics.json 2>/dev/null | wc -l) M1FD001=$(ls experiments/order03/M1_global/FD001/*/metrics.json 2>/dev/null | wc -l)"
for f in $(ls -t logs/order03/M7_*.log 2>/dev/null | head -2); do echo -n "$(basename $f): "; tail -n1 "$f" | cut -c1-60; echo; done
