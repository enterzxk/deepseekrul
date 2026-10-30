
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f '[r]un_candidate') cells=$(ls experiments/order03/*/*/*/metrics.json | wc -l)"
echo "M7_FD002=$(ls experiments/order03/M7_rawback/FD002/*/metrics.json 2>/dev/null | wc -l) M7_FD001=$(ls experiments/order03/M7_rawback/FD001/*/metrics.json 2>/dev/null | wc -l) M6_FD001=$(ls experiments/order03/M6_tfA/FD001/*/metrics.json 2>/dev/null | wc -l) M1_FD001=$(ls experiments/order03/M1_global/FD001/*/metrics.json 2>/dev/null | wc -l)"
