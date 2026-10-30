
cd /home/yndx/zxk/bearing-rul-pi
ls -t logs/order03/split9_M1_*.log 2>/dev/null | head -2 | xargs -r tail -n1
declare -A kept; killed=0
for p in $(pgrep -f "run_candidate.py"); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$cmd" in
    *split9*)
      seed=$(echo "$cmd" | grep -o -- '--seed [0-9]*' | awk '{print $2}')
      if [ -z "${kept[$seed]:-}" ]; then kept[$seed]=$p; else kill -9 $p 2>/dev/null && killed=$((killed+1)); fi;;
    *) kill -9 $p 2>/dev/null && killed=$((killed+1));;
  esac
done
sleep 2
echo "killed=$killed now=$(pgrep -c -f 'run_candidate.py')"
