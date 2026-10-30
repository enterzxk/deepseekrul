
keep=0; killed=0
for p in $(pgrep -f "run_candidate.py"); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$cmd" in
    *split8*) keep=$((keep+1));;
    *) kill -9 $p 2>/dev/null && killed=$((killed+1));;
  esac
done
sleep 2
echo "kept=$keep killed=$killed now=$(pgrep -c -f 'run_candidate.py')"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
