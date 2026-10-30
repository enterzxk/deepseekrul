
cd /home/yndx/zxk/bearing-rul-pi
echo "trainers=$(pgrep -c -f '[r]un_candidate') cells=$(ls experiments/order03/*/*/*/metrics.json | wc -l)"
for f in $(ls -t logs/order03/*.log | head -4); do echo -n "$(basename $f | cut -c1-28) | "; tail -n 1 "$f" | cut -c1-70; echo; done
