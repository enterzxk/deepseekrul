
cd /home/yndx/zxk/bearing-rul-pi/data/CMAPSS
for f in test_FD003.txt RUL_FD003.txt test_FD004.txt RUL_FD004.txt test_FD001.txt RUL_FD001.txt test_FD002.txt RUL_FD002.txt; do
  p=$(stat -c %A "$f")
  chmod 644 "$f"
  n=$(wc -l < "$f")
  c=$(head -1 "$f" | wc -w)
  printf "%-16s %7s lines  %2s cols  %s\n" "$f" "$n" "$c" "$p"
done
chmod 000 test_FD003.txt RUL_FD003.txt test_FD004.txt RUL_FD004.txt
echo "--- after reseal ---"
ls -l test_FD003.txt test_FD004.txt RUL_FD003.txt RUL_FD004.txt | awk '{print $1, $9}'
