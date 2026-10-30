
pids=$(pgrep -f "[o]3d.sh")
echo "controllers: $pids"
for p in $pids; do echo "$p $(ps -o lstart= -p $p)"; done
# keep the oldest, kill the newer duplicates (targeted, not broad pkill)
first=$(echo "$pids" | head -1)
for p in $pids; do
  if [ "$p" != "$first" ]; then kill -9 "$p" 2>/dev/null && echo "killed duplicate controller $p"; fi
done
sleep 2
echo "remaining controllers: $(pgrep -fc '[o]3d.sh')"
echo "trainers: $(pgrep -c -f '[r]un_candidate')"
