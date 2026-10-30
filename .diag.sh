
cd /home/yndx/zxk/bearing-rul-pi
grep -c '^def prepare' scripts/run_candidate.py
grep -c '^def build_cond' scripts/run_candidate.py
ls -l --time-style=+%H:%M:%S scripts/run_candidate.py logs/order01/FD002_A_2027.log
pgrep -c -f run_candidate
tail -n 3 logs/order01/FD002_A_2027.log
