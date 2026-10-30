cd /home/yndx/zxk/bearing-rul-pi
keep=0; kill_n=0
for p in $(pgrep -f "run_candidate.py"); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$cmd" in
    *M3z*) keep=$((keep+1));;
    *) kill -9 $p 2>/dev/null && kill_n=$((kill_n+1));;
  esac
done
sleep 2
echo "kept=$keep killed=$kill_n now=$(pgrep -c -f 'run_candidate.py')"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader