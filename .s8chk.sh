
cd /home/yndx/zxk/bearing-rul-pi
echo "runners=$(pgrep -fc '[s]plit8.sh') trainers=$(pgrep -c -f 'run_candidate.py')"
ls experiments/split8/*/FD002/*/metrics.json 2>/dev/null | wc -l
