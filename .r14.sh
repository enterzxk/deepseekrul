
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f 'run_candidate.py')"
echo "M1_FD001=$(ls experiments/order03/M1_global/FD001/*/metrics.json 2>/dev/null | wc -l) M6_FD001=$(ls experiments/order03/M6_tfA/FD001/*/metrics.json 2>/dev/null | wc -l)"
