
cat /home/yndx/p8/o2b.out 2>/dev/null | head -20
echo "---exists---"
ls -l /home/yndx/p8/o2b.sh
echo "---run---"
cd /home/yndx/zxk/bearing-rul-pi && timeout 20 bash /home/yndx/p8/o2b.sh 2>&1 | head -10
