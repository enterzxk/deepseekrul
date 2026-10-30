
keep=0; killed=0
for p in $(pgrep -f "[r]un_candidate"); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$cmd" in
    *M7_rawback*) keep=$((keep+1));;
    *) kill -9 $p 2>/dev/null && killed=$((killed+1));;
  esac
done
sleep 3
echo "kept_M7=$keep killed=$killed trainers_now=$(pgrep -c -f '[r]un_candidate')"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
