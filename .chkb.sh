
cd /home/yndx/zxk/bearing-rul-pi
echo "runners=$(pgrep -fc '[b]atch.sh') trainers=$(pgrep -c -f '[r]un_candidate')"
ls experiments/order03/M7_rawback/*/*/metrics.json 2>/dev/null | wc -l
ls experiments/order03/M6_tfA/FD001/*/metrics.json 2>/dev/null | wc -l
