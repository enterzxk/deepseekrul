
declare -A kept
killed=0
for p in $(pgrep -f "[r]un_candidate"); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  seed=$(echo "$cmd" | grep -o -- '--seed [0-9]*' | awk '{print $2}')
  if [ -z "${kept[$seed]:-}" ]; then kept[$seed]=$p; else kill -9 $p 2>/dev/null && killed=$((killed+1)); fi
done
sleep 3
echo "killed=$killed trainers_now=$(pgrep -c -f '[r]un_candidate')"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
